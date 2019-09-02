#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     bnc_process.py
@time:     2018-06-01
@function: pre-process the BNC documents and pick out the most common 100000 words in the corpus
"""

import os
from bs4 import BeautifulSoup
import collections
from tqdm import tqdm
import multiprocessing


def process_xml(name, index):
    current_doc = []
    doc_file = BeautifulSoup(open(name, 'rb').read(), 'xml')
    words = doc_file.find_all('w')
    for word in words:
        try:
            current_doc.append(word['hw'])
        except KeyError as error:
            continue
    print(index, name.split('/')[-1], 'done!')
    return current_doc


if __name__ == '__main__':
    file_path_list = list()
    for i in tqdm(os.listdir(r'./2554/2554/download/Texts')):
        for j in os.listdir(r'./2554/2554/download/Texts/%s' % i):
            for k in os.listdir(r'./2554/2554/download/Texts/%s/%s' % (i, j)):
                file_path_list.append(r'./2554/2554/download/Texts/%s/%s/%s' % (i, j, k))

    pool = multiprocessing.Pool(processes=4)
    result = list()

    for d_index, file_name in tqdm(enumerate(file_path_list[:10])):
        result.append(pool.apply_async(process_xml, (file_name, d_index)))
    pool.close()
    pool.join()

    doc_all = list()
    for res in result:
        doc = res.get()
        doc_all.append(doc)

    stopwords = [i.strip() for i in open('./stopwords.txt').readlines()]

    word_dict = collections.Counter()
    for d in tqdm(doc_all):
        word_dict.update(d)

    most_common_words = word_dict.most_common(100000)
    uncommon_words = [i[0] for i in set([i for i in word_dict.most_common()]).difference(set(most_common_words))]
    doc_filter = [[i for i in d if i not in stopwords and i not in uncommon_words] for d in tqdm(doc_all)]

    for d in tqdm(doc_filter):
        open('./hbc_nonstop.txt', 'a+', encoding='utf-8').write(' '.join(d) + '\n')
