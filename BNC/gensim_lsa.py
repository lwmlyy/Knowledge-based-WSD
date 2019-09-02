#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     gensim_lsa.py
@time:     2018-06-08
@function: learn word representation from a corpus composed of general and domain documents
"""

from gensim import models, corpora
import pickle
import numpy as np
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import random


def lsa_vector(data_year='15', domain_doc='all_doc'):
    data_year = data_year
    stwords = [i.strip() for i in open('stopwords.txt').readlines()]
    try:
        gloss = list(set([i.strip() for i in open('./gloss%s.txt' % data_year, 'r').readlines()]))
        context = list(set(
            ' '.join([line.strip() for line in open('./semeval%s.txt' % data_year, 'r').readlines()]).split())) + gloss
    except:
        context = []

    hbc_doc_o = [i.replace('/', ' ').strip().split() for i in
                 open('./BNC/filter_hbc_nonstop.txt', 'r', encoding='gbk').readlines()]

    hbc_doc_add = [[i for i in j.strip().split() if i not in stwords] for j in
                   open('./%s_%s.txt' % (domain_doc, data_year)).readlines()]

    word_dict = collections.Counter()
    for d in tqdm(hbc_doc_add):
        word_dict.update(d)
    most_common_words = [i[0] for i in word_dict.most_common(int(len(word_dict.most_common()) * float(0.8)))]
    uncommon_words = set([i[0] for i in word_dict.most_common()]).difference(set(most_common_words)).difference(
        set(context))
    hbc_doc = [[i for i in d if i not in uncommon_words] for d in tqdm(hbc_doc_add)]

    hbc_doc += hbc_doc_o
    random.shuffle(hbc_doc)

    print('docs ready')
    dictionary = corpora.Dictionary(hbc_doc)
    print(len(dictionary.token2id))

    corpus = [dictionary.doc2bow(text) for text in hbc_doc]

    tf_idf = models.TfidfModel(corpus)  # the constructor
    print('ready to train lsa')
    # this may convert the docs into the TF-IDF space.
    # Here will convert all docs to TFIDF
    corpus_tfidf = tf_idf[corpus]
    # train the lsi model
    topic_num = 200
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)
    matrix = lsi.get_topics()
    topics = lsi.show_topics(num_topics=-1, num_words=100, log=False, formatted=False)
    topic_list = [[j[0] for j in i[1]] for i in topics]
    dict_features = [i[1] for i in sorted(zip(dictionary.token2id.values(), dictionary.token2id.keys()))]
    topic_list_str = [' '.join(i) for i in topic_list]
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(topic_list_str)
    vocab = count_vectorizer.get_feature_names()
    count_sum = np.sum(count_matrix.toarray(), axis=0)

    filter_word = []
    for index, count in enumerate(count_sum):
        if count > topic_num * 0.3:
            filter_word.append(vocab[index])

    topic_list_filter = [list(set(i).difference(set(filter_word)))[:50] for i in topic_list]
    with open('./BNC/eLSA%s' % data_year, 'wb') as fs:
        pickle.dump(matrix, fs, -1)
    with open('./BNC/eLSA_vocab%s' % data_year, 'wb') as fv:
        pickle.dump(dict_features, fv, -1)
    with open('./BNC/eLSA_topic%s' % data_year, 'wb') as fv:
        pickle.dump(topic_list_filter, fv, -1)