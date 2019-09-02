#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     doc_retrieve.py
@time:     2018-05-27
@function: retrieve wikipedia documents based on retrieved most-related doc names
"""

import sqlite3
from main_algorithm import *
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import multiprocessing
import time


# lemmatize all the words with their pos and filter all non-wn words
def doc_filter(doc):
    wnl = WordNetLemmatizer()
    sentences = sent_tokenize(doc)
    doc_filtered = ''
    for sentence in sentences:
        lemmatized_sent = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for
                           i, j in pos_tag(word_tokenize(sentence))]
        wn_filtered = ' '.join([i for i in lemmatized_sent if wn.synsets(i)])
        doc_filtered += wn_filtered + ' '
    doc_filtered = ' '.join(re.sub('[^a-zA-Z]', ' ', doc_filtered).strip().split()).lower()
    return doc_filtered


# save retrieved documents
mydb = sqlite3.connect("path to wikipedia.db")
cursor = mydb.cursor()

data_year = ['01', '04', '07', '13', '15']
names = locals()
doc_names = list()
for year in data_year:
    names['name_dict_%s' % year] = [i.strip().split('\t')[0] for i in open('./docname%s.txt' % year).readlines()]
    doc_names.extend(names['name_dict_%s' % year])
doc_names = list(set(doc_names))


def doc_retrieve(doc_id):
    rest_doc = []
    try:
        cursor.execute("SELECT text FROM documents WHERE id=?", (doc_id,))
        Tables = cursor.fetchall()
        current_doc = doc_filter(Tables[0][0].replace('\n', ' '))
        for year in data_year[:]:
            if doc_id in names['name_dict_%s' % year]:
                with open('./all_doc_%s' % year, 'a+') as fs:
                    fs.write(current_doc + '\n')
    except:
        rest_doc.append(doc_id)
        pass

    return rest_doc


while doc_names:
    t0 = time.time()
    pool = multiprocessing.Pool(processes=8)
    result = list()
    doc_set = dict()
    for index, doc_id in enumerate(tqdm(doc_names)):
        result.append(
            pool.apply_async(doc_retrieve, (doc_id, )))
    pool.close()
    pool.join()
    t1 = time.time()
    print((t1-t0)/60)
    doc_names_rest = []
    for res in result:
        doc = res.get()
        doc_names_rest.extend(doc)
    doc_names = doc_names_rest
