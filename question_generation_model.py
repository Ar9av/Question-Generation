import json
import os

import tensorflow as tf

from question_generation.question_generation_lib.src.demo.instance import \
    AQInstance


class QuestionGenerator():

    def __init__(self):
        FLAGS = tf.app.flags.FLAGS
        model_slug_list = ['RL-S2S-1544356761']
        model_slug_curr = model_slug_list[0]

        question_gen_path = os.path.join(os.path.dirname(__file__), "question_generation_lib/")
        models_base_dir = os.path.join(question_gen_path, "question_gen_models/")
        chkpt_path = os.path.join(models_base_dir, "qgen", model_slug_curr)
        
        with open(chkpt_path+'/vocab.json') as f:
            vocab = json.load(f)

        self.generator = AQInstance(vocab=vocab)
        self.generator.load_from_chkpt(chkpt_path)
