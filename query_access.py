#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     query_access.py
@time:     2018-05-27
@function: retrieve document query from wordnet senses and context
"""

from bs4 import BeautifulSoup
from main_algorithm import *


def fetch_test(data_name):
    test_path = './WSD_Unified_Evaluation_Datasets/%s/%s.data.xml' % (data_name, data_name)

    with open(test_path, 'r') as ft:
        wsd_corpus = ''.join(ft.readlines()).replace('<instance', '<wf').replace('</instance>', '</wf>')
    wsd_bs = BeautifulSoup(wsd_corpus, 'xml')

    text_all = wsd_bs.find_all('text')
    wf_v, all_wf_v = [], []
    for text in tqdm(text_all):
        text_context = []
        for sent in text.find_all('sentence'):
            wfs_wsd = sent.find_all('wf')
            for wf in wfs_wsd:
                wf_v.append((wf['lemma'], wf['pos']))
                text_context.append(wf['lemma'])
        all_wf_v.append(text_context)

    return wf_v, all_wf_v


def gloss_extend_one(o_sense, relation_list):
    extended_list_gloss, extended_list, combine_list = dict(), list(), list()

    for index, relation in enumerate(relation_list):
        combine_list += get_related([o_sense], depth=5, relation=relation) + [(wn.synset(o_sense), wn.synset(o_sense))]

    synset_list = list()
    for sym, o_sym in set(extended_list):
        if sym not in synset_list:
            synset_list.append(sym)
            extended_list_gloss[sym] = retrieve_gloss(sym)
            if sym.name() == '0' or o_sym.name() == '0':
                print(sym, o_sym)

    return extended_list_gloss


# without lemmatization, stemming, and specific symbol filtering
def wn_sense(wf_valuable, save_path):
    """
    for each potential sense from the WSD document, we retrieve its definition and examples of usage as a query,
     also the related synset by the hypernymy and hyponymy relation
    :param wf_valuable: all the wsd words
    :param save_path: save the queries to disk
    :return:
    """
    wn2test = {'a': 'J', 'n': 'N', 'r': 'R', 'v': 'V', 's': 'J'}
    relation_list = ['hypernyms', 'hyponyms']
    allinfo_list = list()
    for wf in tqdm(wf_valuable):
        syn_list = wn.synsets(wf[0])
        for syn in syn_list:
            # retrieved sense must be in the same pos as the query word
            if wn2test[syn.name().split('.')[-2]] in wf[1]:
                allinfo_list.append(' '.join(retrieve_gloss(syn)))
                for e_gloss in gloss_extend_one(syn.name(), relation_list).values():
                    allinfo_list.append(' '.join(e_gloss))

    with open(save_path, 'wb') as fp:
        pickle.dump(allinfo_list, fp, -1)


def wsd_context(all_wf_valuable, save_path, window_size=5, stride=1):
    """
    :param all_wf_valuable: all the context
    :param save_path: save to disk
    :param window_size: window size of the context query
    :param stride: stride of the moving window
    :return:
    """
    context_string = []
    for text_wf_valuable in all_wf_valuable:
        for i in range(window_size, len(text_wf_valuable)-window_size, stride):
            context_string.append(' '.join(text_wf_valuable[i - window_size: i + window_size + 1]))
    with open(save_path, 'wb') as fc:
        pickle.dump(context_string, fc, -1)


if __name__ == '__main__':
    for data_year in ['01', '04', '07', '13', '15']:
        name_dict = {'07': 'semeval2007', '13': 'semeval2013', '15': 'semeval2015', '01': 'senseval2',
                     '04': 'senseval3'}
        sense_save_path = './sense_query%s.txt' % data_year
        context_save_path = './context_query%s.txt' % data_year
        wf_valuable, all_wf_valuable = fetch_test(name_dict[data_year])
        wn_sense(wf_valuable, sense_save_path)
        wsd_context(all_wf_valuable, context_save_path)
