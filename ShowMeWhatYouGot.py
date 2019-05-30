import re
import pickle
import os
import string
import numpy as np
from model_tf import get_cnn_model
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')


class generate_network_ready_files:
    def __init__(self, word_vec_size, max_q_length, max_option_length, max_opt_count, max_sent_para, max_words_sent):
        self.unknown_words_vec_dict = None
        self.unknown_words_vec_dict_file = "unk_word2vec_dict.pkl"

        self.word_vec_size = word_vec_size
        self.max_q_length = max_q_length
        self.max_option_length = max_option_length
        self.max_opt_count = max_opt_count
        self.max_sent_para = max_sent_para
        self.max_words_sent = max_words_sent

    def gen_vec(self, model, raw_data_content, is_closest_para_file=False):
        all_vec_array = np.array([])
        number_of_words = 0

        if is_closest_para_file:
            all_vec_array = np.zeros((self.max_sent_para,self.max_words_sent,self.word_vec_size))
            sents = sent_tokenize(raw_data_content)
            #print(len(sents))
            for i in range(len(sents)):
                words = word_tokenize(sents[i])
                words = [w for w in words if w not in string.punctuation]
                # sanity check
                if len(words)>self.max_words_sent:
                    words = words[:self.max_words_sent]
                for j in range(len(words)):
                    word = words[j].strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array[i,j,:] = vec
        else:            
            words = word_tokenize(raw_data_content)
            words = [w for w in words if w not in string.punctuation]
            for word in words:
                word = word.strip().lower()
                
                vec = self.get_vec_for_word(model, word)
                all_vec_array = np.append(all_vec_array, vec)
                number_of_words += 1
                if number_of_words > self.max_sent_para*self.max_words_sent-1:
                    break

        return all_vec_array

    def get_vec_for_word(self, model, word):
        
        try:
            vec = model[word]
            
            return vec
        except:

            vec = self.handle_unknown_words(word)
            
            return vec

    def handle_unknown_words(self, word):
        fname = self.unknown_words_vec_dict_file
        if self.unknown_words_vec_dict is None:

            if os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    self.unknown_words_vec_dict = pickle.load(f)
            else:
                raise "No dict given"
        if self.unknown_words_vec_dict.get(word, None) is not None:
            vec = self.unknown_words_vec_dict.get(word, None)
        else:

            vec = np.random.rand(1, self.word_vec_size)
            self.unknown_words_vec_dict[word] = vec
        return vec


class ShowMeWhatYouGot:
    def __init__(self, model_weights='sciq_w.h5', vectors='GoogleNews-vectors-negative300.bin.gz'):
        self.model = get_cnn_model()
        self.model.load_weights(model_weights)

        self.vectors = KeyedVectors.load_word2vec_format(vectors, binary=True)

        self.word_vec_size = 300
        self.max_q_length = 68
        self.max_option_length = 12
        self.max_opt_count = 4
        self.max_sent_para = 10
        self.max_words_sent = 25

    def get_answer(self, input_dict):
        exercise = input_dict.copy()

        for key in exercise.keys():
            exercise[key] = exercise[key].encode(
                'ascii', 'ignore').decode('ascii')
            
        exercise['support'] = self.prepare_support(exercise['support'])
        print(exercise['support'])

        gen = generate_network_ready_files(self.word_vec_size, self.max_q_length,
                                           self.max_option_length, self.max_opt_count, 
                                           self.max_sent_para, self.max_words_sent)

        for key in exercise.keys():
            if key == 'support':
                exercise[key] = gen.gen_vec(self.vectors, exercise[key], True)
            else:
                exercise[key] = gen.gen_vec(self.vectors, exercise[key], False)


        options_mat, _ = self.read_options(exercise)

        question_mat = self.read_question(exercise['question'])
        sent_mat = self.read_sentence(exercise['support'])

        answer = self.model.predict([question_mat, sent_mat, options_mat])

        odp, prop = np.argmax(answer), np.max(answer)

        # print(input_dict['question'])
        # print(0, input_dict['answer0'])
        # print(1, input_dict['answer1'])
        # print(2, input_dict['answer2'])
        # print(3, input_dict['answer3'])

        # print('\nOdp: {} (prop: {})\n'.format(odp, prop))

        # print(input_dict['support'])

        return odp, prop

    def prepare_support(self, text):
        raw_data_content = ""
        count = 0
        for s in sent_tokenize(text):
            if len(s.split()) > self.max_words_sent:
                raw_data_content += " ".join(s.split()[:self.max_words_sent])
                raw_data_content += ". "
            else:
                raw_data_content += " ".join(s.split())
                raw_data_content += " "
            count += 1
            if count == self.max_sent_para:
                break

        return raw_data_content


    def read_options(self, exercise):
        complete_array = None
        num_of_options = 0

        for key in ['answer0', 'answer1', 'answer2', 'answer3']:
            complete_array_part = exercise[key]
            complete_array_part = complete_array_part.reshape(
                -1, self.word_vec_size)

            if complete_array_part.shape[0] > self.max_option_length:
                complete_array_part = complete_array_part[:self.max_option_length, :]

            while complete_array_part.shape[0] < self.max_option_length:
                complete_array_part = np.concatenate(
                    (complete_array_part, np.zeros((1, self.word_vec_size))), axis=0)

            complete_array_part = complete_array_part.reshape(
                1, self.max_option_length, self.word_vec_size)

            complete_array = complete_array_part if complete_array is None else np.concatenate(
                (complete_array, complete_array_part), axis=0)
            num_of_options += 1

        num_act_options = num_of_options
        while num_of_options < self.max_opt_count:
            complete_array = np.concatenate((complete_array, np.zeros(
                (1, self.max_option_length, self.word_vec_size))), axis=0)
            num_of_options += 1

        complete_array = complete_array.reshape(
            1, self.max_opt_count, self.max_option_length, self.word_vec_size)

        return complete_array, num_act_options

    def read_question(self, complete_array):
        complete_array = complete_array.reshape(-1, self.word_vec_size)
        if complete_array.shape[0] > self.max_q_length:
            complete_array = complete_array[:self.max_q_length, :]
        while complete_array.shape[0] < self.max_q_length:
            complete_array = np.concatenate(
                (complete_array, np.zeros((1, self.word_vec_size))), axis=0)
        complete_array = complete_array.reshape(
            1, self.max_q_length, self.word_vec_size)
        return complete_array

    def read_sentence(self, complete_array):
        complete_array = np.expand_dims(complete_array, 0)
        return complete_array


if __name__ == '__main__':
    smwyg = ShowMeWhatYouGot(model_weights='sciq_w.h5',
                             vectors='GoogleNews-vectors-negative300.bin.gz')

    import json
    with open('test.json') as json_file:
        data = json.load(json_file)

    data = data[:]

    count = 0
    for i, mydict in enumerate(data):
        mydict['answer0'] = mydict.pop('distractor1')
        mydict['answer1'] = mydict.pop('distractor2')
        mydict['answer2'] = mydict.pop('distractor3')
        mydict['answer3'] = mydict['correct_answer']
        mydict['correct'] = mydict.pop('correct_answer')

        print(i)
        odp, prop = smwyg.get_answer(mydict)
        if odp==3:
            count+=1

    print(count/len(data))
