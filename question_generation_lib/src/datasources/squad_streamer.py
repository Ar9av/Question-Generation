# An abstract class that provides a loader and preprocessor for the SQuAD dataset (or other context/q/a triples)
import numpy as np
import tensorflow as tf

from base_model import TFModel
from helpers.loader import OOV, PAD, EOS, SOS
import helpers.preprocessing as preprocessing

import flags

FLAGS = tf.app.flags.FLAGS

class SquadStreamer():
    def __init__(self, vocab, batch_size, num_epochs=1, shuffle=True):
        self.vocab=vocab
        self.rev_vocab = {v:k for k,v in self.vocab.items()}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs=num_epochs

    def __enter__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_data_pipeline(self.batch_size)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0,visible_device_list="")

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True), graph=self.graph)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    # expects to receive a 4D tuple of squad data as generated by the loader
    def initialise(self, data):
        contexts, qs, answers,a_pos = zip(*data)
        # build ix here - it then refers to the index in the original (unshuffled) dataset
        self.sess.run(self.iterator.initializer, feed_dict={self.context_ph: contexts,
                                          self.qs_ph: qs, self.as_ph: answers, self.a_pos_ph: a_pos,
                                          self.ix: np.arange(len(contexts))})

    def get_batch(self):
        return self.sess.run([self.batch_as_nested_tuple, self.batch_len])


    def build_data_pipeline(self, batch_size):
        with tf.device('/cpu:*'):
            self.context_ph = tf.placeholder(tf.string, [None])
            self.qs_ph = tf.placeholder(tf.string, [None])
            self.as_ph = tf.placeholder(tf.string, [None])
            self.a_pos_ph = tf.placeholder(tf.int32, [None])
            self.ix = tf.placeholder(tf.int32, [None])

            dataset = tf.data.Dataset.from_tensor_slices( (self.context_ph, self.qs_ph, self.as_ph, self.a_pos_ph, self.ix) )

            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=100000)

            # processing pipeline
            dataset = dataset.map(lambda context,q,a,a_pos,ix:
                        (tuple(tf.py_func(preprocessing.process_squad_context(self.vocab, context_as_set=FLAGS.context_as_set), [context], [tf.string, tf.int32, tf.int32, tf.int32, tf.int32])),
                        tuple(tf.py_func(preprocessing.process_squad_question(self.vocab, max_copy_size=FLAGS.max_copy_size, context_as_set=FLAGS.context_as_set, copy_priority=FLAGS.copy_priority, smart_copy=FLAGS.smart_copy, latent_switch=FLAGS.latent_switch), [q,context,a_pos], [tf.string, tf.int32, tf.float32, tf.int32])),
                        tuple(tf.py_func(preprocessing.process_squad_answer(self.vocab, context_as_set=FLAGS.context_as_set), [a,a_pos,context], [tf.string, tf.int32, tf.int32, tf.int32])),
                        ix
                        # q,a
                        ))



            # pad out to batches
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                                tf.TensorShape([None]),  # source vectors of unknown size
                                tf.TensorShape([None]),
                                tf.TensorShape([]),      # size(source)
                                tf.TensorShape([])),     # size(source vocab)
                               (tf.TensorShape([None]),  # target vectors of unknown size
                                tf.TensorShape([None]),  # target vectors of unknown size
                                tf.TensorShape([None, None]),  # target vectors of unknown size
                                tf.TensorShape([])),     # size(source)
                               (tf.TensorShape([None]),  # target vectors of unknown size
                                tf.TensorShape([None]),  # target vectors of unknown size
                                tf.TensorShape([]),
                                tf.TensorShape([None])),    # size(target)
                                tf.TensorShape([])), #ix
                padding_values=((PAD,
                                self.vocab[PAD],  # source vectors padded on the right with src_eos_id
                                 0,
                                 len(self.vocab),
                                 0),          # size(source) -- unused
                                (PAD,
                                self.vocab[PAD],  # target vectors padded on the right with tgt_eos_id
                                 0.0,
                                 0),          # size(source) -- unused
                                (PAD,
                                self.vocab[PAD],  # target vectors padded on the right with tgt_eos_id
                                 0, # answer len
                                 0),# answer locs
                                 0)) # ix

            dataset = dataset.repeat(self.num_epochs)

            dataset = dataset.prefetch(buffer_size=batch_size*4)

            self.iterator = dataset.make_initializable_iterator()
            self.batch_as_nested_tuple = self.iterator.get_next()
            self.this_context, self.this_question, self.this_answer, self.this_ix = self.batch_as_nested_tuple
            (self.context_raw, self.context_ids, self.context_copy_ids, self.context_length, self.context_vocab_size) = self.this_context
            (self.question_raw, self.question_ids, self.question_oh, self.question_length) = self.this_question
            (self.answer_raw, self.answer_ids, self.answer_length, self.answer_locs) = self.this_answer

            self.batch_len = tf.shape(self.context_raw)[0]
