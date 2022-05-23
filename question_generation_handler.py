import json
import os
import pickle
import string
import sys
import uuid

import tensorflow as tf
from nltk import PorterStemmer
from nltk.corpus import wordnet as wn

from utility.config.app_config import LANGUAGES
from utility.helpers.network_helper import RemoteModel, RemoteQuestionGenerator
from utility.helpers.utils import Timer
from utility.nlp.text_normalizer import TextNormalizer
from utility.utils.api_utils import APIHelper
from generic_entity_extractor import GenericEntityExtractor
from logger import Logger
from question_generation.question_generation_lib.src.demo.instance import \
    AQInstance
from question_generation.question_generation_lib.src.helpers import \
    preprocessing
from question_generation.question_generation_model import QuestionGenerator

LOGGER = Logger(__name__)

question_gen_path = os.path.join(os.path.dirname(__file__), "question_generation_lib/")
models_base_dir = os.path.join(question_gen_path, "question_gen_models/")
data_base_dir = os.path.join(question_gen_path, "question_gen_data/")

GENERATOR = QuestionGenerator().generator

generic_entity_extractor = GenericEntityExtractor(should_load_spacy=True)

class QuestionGenerationHandler():

    def __init__(self):
        self.remote_BERT_answering_batch_size = 10
        self.BERT_answers_threshold = 0.5
        self.remote_question_gen_batch_size = 5

    def question_generator_wrapper(self, chunks_list):
        """
        Args:
            - chunks_list (list): list of dictionaries with each dict having keys - "text", "answers", "chunk_id"
        """
        generated_questions = self.generate_questions(
            chunks_list, 
            batch_size=self.remote_question_gen_batch_size
            )
        generated_questions = [ques for ques in generated_questions if ques]
        filtered_questions = []

        if generated_questions:
            filtered_questions = self.clean_up_questions(generated_questions)
            # filtered_questions = self.get_similar_questions(filtered_questions)

            LOGGER.info(f"QuestionGenerationHandler.question_generator_wrapper: ------- "
                        f"Generated {len(filtered_questions)} questions -------"
            )
            LOGGER.info(f"QuestionGenerationHandler.question_generator_wrapper: ------- "
                        "Completed question generation -------"
            )
        else:
            LOGGER.info(f"QuestionGenerationHandler.question_generator_wrapper: No questions generated")
            raise ValueError('0 QUESTION GENERATED, quitting KP VALIDATION')

        return filtered_questions
    
    def generate_questions(self, qa_list, batch_size=5):
        lap_timer = Timer()
        questions_generated = []
        questions_generated = RemoteQuestionGenerator.generate_questions(qa_list, batch_size=batch_size)
        LOGGER.info(f"QuestionGenerationHandler.generate_questions: Generated {len(questions_generated)}"
                    f" questions in time {lap_timer.lap()}")
        return questions_generated
    
    def generate_questions_remote(self, qa_list):
        questions_generated = []
        for i, qa_pair in enumerate(qa_list):
            text = qa_pair["text"].lower()
            answers = qa_pair["answers"]
            chunk_id = qa_pair["chunk_id"]
            for answer in answers:
                answer = answer.lower()
                answer_pos = int(text.find(answer))
                question = GENERATOR.get_q(text.encode(), answer.encode(), answer_pos)
                questions_generated.append(
                    {
                    "question": question, 
                    "context": text, 
                    "id": str(uuid.uuid4()), 
                    "chunk_id": chunk_id
                    }
                    )
        return questions_generated
    
    def clean_up_questions(self, questions_generated):
        time_dict = {}
        lap_timer = Timer()
        questions_generated = self.basic_clean_up_questions(questions_generated)
        time_dict["basic_clean_up"] = lap_timer.lap()

        LOGGER.info(f"QuestionGenerationHandler.clean_up_questions: Fetching answers from BERT....")
        
        all_predictions, all_n_best_predictions, _ = \
            self.get_answers_from_bert(
                questions_generated, 
                batch_size=self.remote_BERT_answering_batch_size
                )

        time_dict["BERT_answers_received"] = lap_timer.lap()
        LOGGER.info(f"QuestionGenerationHandler.clean_up_questions: Got BERT answers")

        answered_questions = self.get_answered_bert_questions\
            (questions_generated, all_predictions, all_n_best_predictions)

        time_dict["probability_filtered_BERT_answers"] = lap_timer.lap()
        LOGGER.info(f"QuestionGenerationHandler.clean_up_questions: Filtered BERT answers by probability")

        filtered_questions = self.remove_non_questions(answered_questions)
        filtered_questions = self.remove_special_characters(filtered_questions)

        time_dict["special_character_filtering"] = lap_timer.lap()

        filtered_questions = self.remove_non_noun_questions(filtered_questions)

        LOGGER.info(f"QuestionGenerationHandler.clean_up_questions: Time taken to clean up questions ---->>> {time_dict}")
        
        return filtered_questions

    def basic_clean_up_questions(self, questions_generated):
        for question in questions_generated:
            try:
                question["question"] = question["question"].decode('utf-8')
            except:
                pass
            try:
                pos = question["question"].find("?")
                question["question"] = question["question"][:(pos+1)]
            except:
                pass
        
        return questions_generated

    def get_answers_from_bert(self, questions_generated, batch_size=20):
        batches = []
        batch = 0
        all_predictions = []
        all_n_best_predictions = []
        all_scores_diff = []
        while batch <= len(questions_generated):
            batches.append(questions_generated[batch:batch+batch_size])
            batch += batch_size
            predictions, n_best_predictions, scores_diff = \
                RemoteModel.predict_parallel(query_examples=questions_generated)
            all_predictions.extend(predictions)
            all_n_best_predictions.extend(n_best_predictions)
            all_scores_diff.extend(scores_diff)
        
        return all_predictions, all_n_best_predictions, all_scores_diff
    
    def get_custom_data(self, questions_generated):
        custom_data = {}
        for question in questions_generated:
            custom_data[question["id"]] = question
        
        return custom_data

    def get_answered_bert_questions(self, questions_generated, all_predictions, all_n_best_predictions, threshold = 0.5):
        custom_data = self.get_custom_data(questions_generated)
        n_best_pred = all_n_best_predictions[0]
        answers_to_questions = {}
        prob_filtered_data = []

        for key, val in all_predictions[0].items(): 
            if val != '': 
                answers_to_questions[key] = val 
        
        for key, val in answers_to_questions.items():
            prob = n_best_pred[key][0]['probability']
            if prob >= threshold:
                temp_val = custom_data[key]
                temp_val["answer"] = val
                prob_filtered_data.append(temp_val)
            
        return prob_filtered_data
    
    def remove_non_questions(self, prob_filtered_data, language=LANGUAGES.EN):
        ques_word = ['what', 'when', 'why', 'who', 'how', 'where', 'can', 'whom']
        text_normalizer = TextNormalizer(language)

        for question in prob_filtered_data:
            ques = question["question"]
            ques_tokens = text_normalizer.tokenize(ques)
            if ques_tokens[0] in ques_word or ques_tokens[len(ques_tokens)-2] in ques_word:
                question["question"] = ques
            else:
                question = None
        filtered_questions = [x for x in prob_filtered_data if x is not None]
        return filtered_questions
    
    def remove_special_characters(self, prob_filtered_data):

        for question in prob_filtered_data:
            if not self.is_ascii(question["question"]):
                question = None
        
        filtered_questions = [x for x in prob_filtered_data if x is not None]
        return filtered_questions
    
    def is_ascii(self, s):
        return all(ord(c) < 128 for c in s) 
    
    def remove_non_noun_questions(self, prob_filtered_data):

        for question in prob_filtered_data:
            doc = generic_entity_extractor.nlp(question["question"])
            tokens_list = [x.pos_ for x in doc]
            if "NOUN" not in tokens_list:
                question = None
        
        filtered_questions = [x for x in prob_filtered_data if x is not None]
        return filtered_questions
    
    def get_similar_questions(self, filtered_questions):

        for question in filtered_questions:
            ques = generic_entity_extractor.nlp(question["question"])
            ques_tokens = [x for x in ques]
            similar_questions = []
            for i, token in enumerate(ques_tokens):
                if token.pos_ == "NOUN":
                    try:
                        tmp_word = token.text
                        syn = wn.synsets(token.text, pos=wn.NOUN)[0]
                        word_replacement = syn.lemmas()[0].name()
                        if tmp_word != word_replacement:
                            ques_tokens[i] = word_replacement
                            print(f"replaced {tmp_word} with {word_replacement}")
                    except:
                        pass
                else:
                    pass
            ques_tokens = [str(x) for x in ques_tokens]
            if " ".join(ques_tokens) != question["question"]:
                similar_questions.append(" ".join(ques_tokens))
            ques_tokens = [x for x in ques]
            for i, token in enumerate(ques_tokens):
                if token.pos_ == "VERB":
                    try:
                        tmp_word = token.text
                        syn = wn.synsets(token.text, pos=wn.VERB)[0]
                        word_replacement = syn.lemmas()[0].name()
                        if tmp_word != word_replacement:
                            ques_tokens[i] = word_replacement
                            print(f"replaced {tmp_word} with {word_replacement}")
                    except:
                        pass
                else:
                    pass
            ques_tokens = [str(x) for x in ques_tokens]
            if " ".join(ques_tokens) != question["question"]:
                similar_questions.append(" ".join(ques_tokens))
            question["similar_questions"] = similar_questions

        return filtered_questions
