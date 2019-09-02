#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     disambiguation.py
@time:     2018-06-15
@function: disambiguate the document using the similarity of sense vector and context vector
"""

from My_WSD.main_algorithm import *
from bs4 import BeautifulSoup
from My_WSD.BNC.gensim_lsa import *
import time
import multiprocessing
from optparse import OptionParser

usage = "disambiguation -l <learn_emb_bool> -d <domain_doc_name>"
parser = OptionParser(usage, version="%prog 1.0")
parser.add_option("-l", "--learn_emb", dest="learn_emb_bool", default=False,
                  help="whether to learn embeddings from a corpus", metavar="learn_emb_bool")
parser.add_option("-d", "--domain_doc", dest="domain_doc_name", default='domain_doc',
                  help="name of the domain knowledge document", metavar="domain_doc_name")
(options, args) = parser.parse_args()


def get_semeval(data_year):
    if data_year == '01':
        test_path = './WSD_Unified_Evaluation_Datasets/senseval2/senseval2.data.xml'
        gold_path = './WSD_Unified_Evaluation_Datasets/senseval2/senseval2.gold.key.txt'
    elif data_year == '04':
        test_path = './WSD_Unified_Evaluation_Datasets/senseval3/senseval3.data.xml'
        gold_path = './WSD_Unified_Evaluation_Datasets/senseval3/senseval3.gold.key.txt'
    elif data_year == '07':
        test_path = './WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        gold_path = './WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
    elif data_year == '13':
        test_path = './WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.data.xml'
        gold_path = './WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt'
    elif data_year == '15':
        test_path = '../My_WSD/WSD_Unified_Evaluation_Datasets/semeval2015/semeval2015.data.xml'
        gold_path = '../My_WSD/WSD_Unified_Evaluation_Datasets/semeval2015/semeval2015.gold.key.txt'
    else:
        test_path = ''
        gold_path = ''
        print('no data for such year.')

    gold_keys = open(gold_path, 'r').readlines()
    gold_dict = dict([(i.strip().split(' ')[0], i.strip().split(' ')[1:]) for i in gold_keys])
    print(len(gold_dict.items()))

    with open(test_path, 'r') as ft:
        wsd_corpus = ''.join(ft.readlines()).replace('<instance', '<wf').replace('</instance>', '</wf>')
    wsd_bs = BeautifulSoup(wsd_corpus, 'xml')

    text_all = wsd_bs.find_all('text')

    doc_all, context_all = list(), list()
    id_count = 0
    text_context = ''
    for text in tqdm(text_all):
        sent_all = list()
        # text_context = ' '.join(' '.join(text.text.split('\n')).split())
        for sent in text.find_all('sentence'):
            wfs_wsd = sent.find_all('wf')
            lemma_dict = dict()
            id_count_sent = 0
            for wf in wfs_wsd:
                try:
                    lemma_dict[wf['id']] = (id_count_sent, wf['lemma'], wf['pos'], wf['id'])
                except:
                    lemma_dict[id_count] = (id_count_sent, wf['lemma'], wf['pos'], str(id_count_sent))
                    id_count += 1
                text_context += wf['lemma'] + ' '
                id_count_sent += 1
            sent_all.append(lemma_dict)
        doc_all.append(sent_all)
        context_all.append(text_context)

    return doc_all, context_all, gold_dict, len(gold_keys)


def doc_disambiguation(document, gold, data_year):
    correct, correct_p, count_dis = 0, 0, 0
    name_dict = {'07': 'semeval2007', '13': 'semeval2013', '15': 'semeval2015', '01': 'senseval2',
                 '04': 'senseval3'}
    for sent_index, sent_dict in enumerate(tqdm(document)):
        graph_set = [(i[1], i[2]) for i in sorted(sent_dict.values())]
        wsd_id = sorted([w_id for w_id in list(set(gold.keys()).intersection(set(sent_dict.keys())))])
        wsd_index = [graph_set.index(j) for j in [(sent_dict[i][1], sent_dict[i][2]) for i in wsd_id]]
        if wsd_index:
            count_dis += len(wsd_id)
            length = 3
            expand_doc = [{}] * length + document + [{}] * length
            three_sentence = sum([[i[1] for i in sent_dict.values()] for sent_dict in
                                  expand_doc[sent_index:sent_index + 1 + 2 * length] if sent_dict], [])
            similarity_list, similarity_list_path = graph_construct(graph_set, wsd_index, three_sentence, data_year)
            for sim_index, similarity_final in enumerate(similarity_list):

                max_pos = np.where(similarity_final == np.max(similarity_final))[0][0]
                chosen_sense = \
                    retrieve_sense(sent_dict[wsd_id[sim_index]][1], sent_dict[wsd_id[sim_index]][2])[1][
                        max_pos]

                sense_lemmas = [i.key() for i in chosen_sense.lemmas()]
                intersect = set(sense_lemmas).intersection(set(gold[wsd_id[sim_index]]))

                if intersect:
                    correct += 1

            for s_index, similarity in enumerate(similarity_list_path):
                chosen_sense = \
                    retrieve_sense(sent_dict[wsd_id[s_index]][1], sent_dict[wsd_id[s_index]][2])[1][
                        np.where(similarity == np.max(similarity))[0][0]]

                sense_lemmas = [i.key() for i in chosen_sense.lemmas()]
                intersect = set(sense_lemmas).intersection(set(gold[wsd_id[s_index]]))

                if intersect:
                    correct_p += 1

                synset_offset = str(chosen_sense.offset()).zfill(8)
                pos_ = chosen_sense.name().split('.')[-2].replace('s', 'a')
                offset_pos = '%s-%s' % (synset_offset, pos_)
                open('./raw.KWSD.key', 'a+').write('CTX001' + ' ' + '%s.%s' % (
                    name_dict[data_year], sent_dict[wsd_id[s_index]][3]) + ' ' + offset_pos + ' !! ' +
                                                  sent_dict[wsd_id[s_index]][1] + '\n')

    return correct, correct_p, count_dis


if __name__ == '__main__':
    hyper_list, hypo_list = list(), list()
    for data_year in ['01', '04', '13', '15', '07']:
        time_start = time.time()

        if options.learn_emb_bool:
            lsa_vector(data_year, options.domain_doc_name)
            vector_compress(data_year)

        correct_count, correct_count_path, disam = 0, 0, 0

        doc_set, context_set, gold_labels, wsd_all = get_semeval(data_year)

        pool = multiprocessing.Pool(processes=4)
        result = list()
        for d_index, doc in enumerate(doc_set[:]):
            result.append(pool.apply_async(doc_disambiguation, (doc, gold_labels, data_year)))
        pool.close()
        pool.join()
        for res in result:
            c, cp, d = res.get()
            correct_count += c
            correct_count_path += cp
            disam += d

        recall = float(correct_count / wsd_all)
        precision = float(correct_count / disam)
        f1 = float(2 * recall * precision) / (precision + recall)
        print('F1: %s' % str(correct_count_path/wsd_all))
        print(correct_count_path, disam, len(gold_labels.items()), wsd_all)
        time_end = time.time()
        print('Time consumed: %s mins' % str((time_end-time_start)/60))
