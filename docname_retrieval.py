#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     docname_retrieval.py
@time:     2018-05-27
@function: retrieve document names based on sense and context query
"""

import argparse
import logging
from drqa import retriever
import pickle
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=20):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    # print(doc_names, doc_scores)
    retrieved_dict = dict()
    if doc_names:
        for i in range(len(doc_names)):
            if doc_scores[i] >= 100:
                retrieved_dict[doc_names[i]] = doc_scores[i]
    return retrieved_dict


if __name__ == '__main__':
    for data_year in ['01', '04', '07', '13', '15'][:]:
        sense_save_path = './sense_query%s.txt' % data_year
        context_save_path = './context_query%s.txt' % data_year
        doc_name_path = './docname%s.txt' % data_year
        with open(sense_save_path, 'rb') as fs:
            sense_query = pickle.load(fs)
        with open(context_save_path, 'rb') as fc:
            all_query = pickle.load(fc) + sense_query
        doc_set = dict()
        for q in tqdm(all_query[:]):
            doc_set.update(process(q))
        with open(doc_name_path, 'a+') as fd:
            for doc_n, doc_s in doc_set.items():
                fd.write(str(doc_n) + '\t' + str(doc_s) + '\n')
