import os
from os import listdir
from os.path import join, isfile, isdir
import numpy as np
from utilities import Constants
import math


def get_file_url_list(parent_dir, file_type='pdf', file_url_list=None):
    if file_url_list is None:
        file_url_list = list()

    for f in listdir(parent_dir):
        temp_dir = join(parent_dir, f)
        if isfile(temp_dir) and temp_dir.endswith(file_type):
            file_url_list.append(temp_dir)
        elif isdir(temp_dir):
            get_file_url_list(temp_dir, file_type, file_url_list)
    return file_url_list


def is_all_chinese(token_str):
    for _char in token_str:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def remove_stop_words(word_list):
    if Constants.ENABLE_STOP_WORDS:
        return [word for word in word_list if word not in Constants.STOP_WORDS]
    else:
        return word_list


def get_idf(doc_num, tf_inverted_index=None):
    idf = dict.fromkeys(tf_inverted_index.keys(), 0.0)
    for term, doc_num_with_term in tf_inverted_index.items():
        n_i = len(doc_num_with_term)
        idf_i = math.log((doc_num - n_i + 0.5) / (n_i + 0.5))
        idf[term] = idf_i
    return idf


def cosine_similarity(vec_x, vec_y):
    num = np.dot(vec_x, vec_y)
    if num == 0.0:
        return 0.0
    similarity = num / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))
    return similarity


def word_similarity(word_1, word_2, word2vec_model):
    if word_1 not in word2vec_model or word_2 not in word2vec_model:
        return 0.0
    vec1 = word2vec_model[word_1]
    vec2 = word2vec_model[word_2]
    similarity = cosine_similarity(vec1, vec2)
    return similarity


def word_document_similarity(word, doc_seg_list, word2vec_model, k4=0.4):
    scores = []
    length = len(doc_seg_list)
    for index in range(length):
        score = word_similarity(word, doc_seg_list[index], word2vec_model)
        # k4控制位置对权重的影响，（0,1）影响越来越大
        score *= (1.0 + (1.0 - (index / length)) * k4)
        scores.append(score)
    return np.mean(scores)


def debug_print(debug_info, mode=Constants.DEBUG_MODE):
    if mode:
        print(debug_info)


if __name__ == '__main__':
    my_parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_file_url_list = get_file_url_list(my_parent_dir)
    print(my_parent_dir)
    print(my_file_url_list)
