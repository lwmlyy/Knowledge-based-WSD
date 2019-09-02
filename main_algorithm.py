#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version:  1.0
@author:   @@@
@file:     main_algorithm.py
@time:     2018-06-11
@function: conduct similarity and graph-based method of WSD
"""

from nltk.corpus import wordnet as wn
from tqdm import tqdm
import re
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import time


def retrieve_lsa(data_year):
    lsa_mat_path = './eLSA%s' % data_year
    lsa_vocab_path = './eLSA_vocab%s' % data_year
    lsa_matrix = pickle.load(open(lsa_mat_path, 'rb'))
    lsa_vocab = pickle.load(open(lsa_vocab_path, 'rb'))

    return lsa_matrix, lsa_vocab


def retrieve_gloss(sense):
    wnl = WordNetLemmatizer()
    lemmatized_sent = ' '.join(
        [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for i, j in
         pos_tag(word_tokenize(sense.definition()))])
    definition = re.sub('[^a-zA-Z]', ' ', lemmatized_sent).lower().split()

    # remove _ between connected phrases and repeated words
    lemma_names = list(set(re.sub('[^a-zA-Z]', ' ', ' '.join(sense.lemma_names())).lower().split()))

    examples = re.sub('[^a-zA-Z]', ' ', ' '.join([' '.join(
        [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for i, j in
         pos_tag(word_tokenize(example))]) for example in sense.examples()])).lower().split()

    return definition + lemma_names + examples


def retrieve_sense(word, pos=None):
    """
        retrieve sense glosses, sense inventory and sense frequency of word as a dict, list and list respectively
    """
    fre_list = list()
    wn2test = {'a': 'ADJ', 'n': 'NOUN', 'r': 'ADV', 'v': 'VERB', 's': 'ADJ'}
    sense_inventory = [i for i in wn.synsets(word) if wn2test[i.name().split('.')[-2]] in pos]

    gloss_list, sense_inventory_final = list(), list()
    for sense in sense_inventory:
        lemma_names = [i.name().lower() for i in sense.lemmas()]
        if word.lower() in lemma_names:
            name = sense.name()
            gloss_list.append((name, retrieve_gloss(sense)))
            sense_inventory_final.append(sense)
            fre_list.append(sense.lemmas()[lemma_names.index(word.lower())].count())
    return gloss_list, sense_inventory_final, fre_list


def get_related(names, depth=15, relation='hypernyms', r_list=None):
    """
    this can be implemented with wn.closure instead, but lower efficiency
    :param names: the synset list
    :param depth: how deep the relation will dive
    :param relation: all the relations
    :param r_list: the (synset, gloss) list from the last iteration
    :return: the extended gloss list with its according synset name
    """
    if not r_list:
        relation_list = list()
    else:
        relation_list = r_list
    if not names or depth == 0:
        return relation_list
    else:
        current_hypernyms = list()
        for name in names:
            if relation == 'hypernyms':
                wn_relation = wn.synset(name).hypernyms()
            elif relation == 'hyponyms':
                wn_relation = wn.synset(name).hyponyms()
            elif relation == 'part_holonyms':
                wn_relation = wn.synset(name).part_holonyms()
            elif relation == 'part_meronyms':
                wn_relation = wn.synset(name).part_meronyms()
            elif relation == 'member_holonyms':
                wn_relation = wn.synset(name).member_holonyms()
            elif relation == 'member_meronyms':
                wn_relation = wn.synset(name).member_meronyms()
            elif relation == 'entailments':
                wn_relation = wn.synset(name).entailments()
            elif relation == 'attributes':
                wn_relation = wn.synset(name).attributes()
            elif relation == 'antonyms':
                wn_relation = [i.synset() for i in wn.synset(name).lemmas()[0].antonyms()]
            else:
                print('no such relation, process terminated.')
                break
            current_hypernyms += [i.name() for i in wn_relation]
            relation_list += [(i, wn.synset(name)) for i in wn_relation]

        return get_related(list(set(current_hypernyms)), depth-1, relation, r_list=relation_list)


def morpho_extend(extended_list):
    morpho_list = list()
    for synset in extended_list:
        morpho_related = [j.synset() for j in
                          list(sum([i.derivationally_related_forms() for i in synset.lemmas()], []))]
        morpho_list.extend([(i, synset) for i in morpho_related])

    return morpho_list


def gloss_extend(o_sense, relation_list):
    extended_list_gloss, extended_list, combine_list = dict(), list(), list()

    # expand the original sense with nearby sense, which turns out to damage the performance
    for index, relation in enumerate(relation_list[1:]):
        combine_list += get_related([o_sense], depth=1, relation=relation) + [(wn.synset(o_sense), wn.synset(o_sense))]

    # expand the expanded list with in-depth hypernyms
    extended_list += get_related([i[0].name() for i in list(set(combine_list))]) + list(set(combine_list))

    extended_list += morpho_extend([i[0] for i in list(set(extended_list))])

    synset_list = list()
    for sym, o_sym in set(extended_list):
        if sym not in synset_list:
            synset_list.append(sym)
            extended_list_gloss[(sym.name(), o_sym.name())] = retrieve_gloss(sym)

    return extended_list_gloss


def compute_tfigf(extended_glosses):
    """
        compute the inverse gloss(extended) frequency(similar to IDF), currently, TF-IGF instead.
        we treat each extended gloss as a document and compute the IGF of each word in the gloss
    """
    IGF_vectorizer = TfidfVectorizer(stop_words='english')
    IGF_matrix = IGF_vectorizer.fit_transform(extended_glosses)
    vocab = IGF_vectorizer.get_feature_names()

    return IGF_matrix.toarray(), vocab


def compute_distance(all_relation, tfigf):
    """
    note: those words occur in different glosses should be weighted according to
    the nearest distance to the original synset, implemented with sorted()
    :param all_relation: a list contains all synset-gloss dicts
    :param tfigf: the tfidf matrix, mainly for its shape and vocab
    :return:
    """
    distance_matrix = np.ones(tfigf[0])
    for index, relation in enumerate(all_relation):
        o_sense = relation[0]
        sorted_gloss = sorted([(1 / (1 + wn.synset(synset).shortest_path_distance(wn.synset(o_sense)))
                                if wn.synset(synset).shortest_path_distance(wn.synset(o_sense)) is not None
                                else 1 / (1 + 5), gloss) for (synset, o_synset), gloss in relation[1].items()])
        for value, e_gloss in sorted_gloss:
            gloss_index = [tfigf[1].index(i) for i in e_gloss if i in tfigf[1]]
            distance_matrix[index, gloss_index] = value

    return distance_matrix


def compute_vector(gloss_extended, tfigf_vocab, distance_vec, data_year):
    lsa_matrix, lsa_vocab = retrieve_lsa(data_year)

    intersect_list = [i for i in list(
        set(gloss_extended.lower().split()).intersection(set(lsa_vocab)).intersection(tfigf_vocab))]
    gloss_extended_filtered = sorted([i for i in gloss_extended.lower().split() if i in intersect_list])
    gloss_lsa_index = [lsa_vocab.index(i) for i in gloss_extended_filtered]
    gloss_matrix_lsa = lsa_matrix[:, gloss_lsa_index]
    gloss_tfigf_index = [tfigf_vocab.index(i) for i in gloss_extended_filtered]
    gloss_distance = np.reshape(distance_vec, (1, distance_vec.shape[0]))[:, gloss_tfigf_index]
    gloss_vector_lsa = np.dot(gloss_matrix_lsa, gloss_distance.T)
    gloss_vector = gloss_vector_lsa
    return gloss_vector.T


def sense_keys(word):
    sense_inventory = wn.synsets(word)
    sense_key = [[i.key() for i in synset.lemmas()] for synset in sense_inventory]
    return sense_key


def sense_matrix(word, word_pos):
    close_list = ['hypernyms', 'hyponyms', 'part_holonyms', 'part_meronyms', 'member_holonyms',
                  'member_meronyms', 'entailments', 'attributes']
    all_sense_list, _, sense_frequency = retrieve_sense(word, word_pos)
    all_relation_list = list()
    for sense_gloss in all_sense_list:
        gloss_dict = dict()
        gloss_dict.update(gloss_extend(sense_gloss[0], close_list))
        all_relation_list.append((sense_gloss[0], gloss_dict))

    return all_relation_list, sense_frequency


def retrieve_xwn():
    noun_xwn = json.load(open('./XWN-2.1/noun_xwn.json'))
    verb_xwn = json.load(open('./XWN-2.1/verb_xwn.json'))
    adj_xwn = json.load(open('./XWN-2.1/adj_xwn.json'))
    adv_xwn = json.load(open('./XWN-2.1/adv_xwn.json'))
    return noun_xwn, verb_xwn, adj_xwn, adv_xwn


def graph_construct(sentence, wsd_index, three_sentence, data_year):
    name = locals()
    name['n_xwn'], name['v_xwn'], name['a_xwn'], name['r_xwn'] = retrieve_xwn()
    relation_list_all, frequency_list_all, similarity_list, similarity_list_path = list(), list(), list(), list()
    sense_length, vi_pair_all, max_index_all, distance_matrix = list(), list(), list(), list()

    for word, pos in sentence:
        if retrieve_sense(word, pos)[1]:
            relation, frequency = sense_matrix(word, pos)
            frequency = np.array(frequency)
            frequency = (frequency + 1) / (frequency.sum() + frequency.shape[0])
            relation_list_all.append(relation)
            frequency_list_all.append(frequency)
            max_index_all.append([i for i in range(len(frequency))])
        else:
            relation_list_all.append([])
            frequency_list_all.append([])
            max_index_all.append([])

    sent_context_vec, topic_num, _ = context_vector(three_sentence, data_year)
    all_sense_vec = np.zeros((1, topic_num))

    for index in wsd_index:

        pure_ex_gloss = [' '.join(sum(d[1].values(), [])) for d in relation_list_all[index]]
        t0 = time.time()
        tfigf = compute_tfigf(pure_ex_gloss)
        distance_mat = compute_distance(relation_list_all[index], (tfigf[0].shape, tfigf[1]))
        t1 = time.time()

        sense_mat = np.zeros((len(relation_list_all[index]), topic_num))
        for sense_index, sense_gloss in enumerate(relation_list_all[index]):
            sense_mat[sense_index, :] = compute_vector(pure_ex_gloss[sense_index], tfigf[1],
                                                       distance_mat[sense_index], data_year)

        similarity = np.dot(sense_mat, sent_context_vec)
        similarity_final = (similarity.T * frequency_list_all[index]).T
        similarity_list.append(similarity_final)
        vi_pair = sorted([(value, index) for index, value in enumerate(similarity_final / similarity_final.sum())],
                         reverse=True)[:3]

        max_index = [i[1] for i in vi_pair]
        sense_mat_filtered = sense_mat[max_index]
        all_sense_vec = np.concatenate((all_sense_vec, sense_mat_filtered), axis=0)
        sense_length.append(len(max_index))
        vi_pair_all.append(vi_pair)
        max_index_all[index] = max_index

    # add the sense vector of those words need not to be disambiguated
    sense_length_non, vi_pair_all_non = list(), list()
    for r_index, word_relation in enumerate(relation_list_all):
        if r_index not in wsd_index and word_relation:
            pure_ex_gloss = [' '.join(sum(d[1].values(), [])) for d in word_relation]
            tfigf = compute_tfigf(pure_ex_gloss)
            distance_mat = compute_distance(relation_list_all[r_index], (tfigf[0].shape, tfigf[1]))
            sense_mat = np.zeros((len(relation_list_all[r_index]), topic_num))
            for sense_index, sense_gloss in enumerate(relation_list_all[r_index]):
                sense_mat[sense_index, :] = compute_vector(pure_ex_gloss[sense_index],
                                                           tfigf[1], distance_mat[sense_index], data_year)
            similarity_non = np.dot(sense_mat, sent_context_vec)
            similarity_final_non = (similarity_non.T * frequency_list_all[r_index]).T
            vi_pair_non = sorted(
                [(value, index) for index, value in enumerate(similarity_final_non / similarity_final_non.sum())],
                reverse=True)[:3]
            max_index_non = [i[1] for i in vi_pair_non]
            sense_mat_filtered = sense_mat[max_index_non]
            all_sense_vec = np.concatenate((all_sense_vec, sense_mat_filtered), axis=0)
            sense_length_non.append(len(max_index_non))
            vi_pair_all_non.append(vi_pair_non)
            max_index_all[r_index] = max_index_non

    sim_matrix = all_sense_vec[1:].dot(all_sense_vec[1:].T)

    sim_matrix[sim_matrix < 0] = 0
    sim_matrix[sim_matrix != 0] = 1

    value_vector = np.ones((sim_matrix.shape[0], 1)) / sim_matrix.shape[0]
    end_index = 0
    for m_index, ws_index in enumerate(wsd_index):
        sim_matrix[end_index:end_index + sense_length[m_index], end_index:end_index + sense_length[m_index]] = 0
        value_vector[end_index:end_index + sense_length[m_index]] = [i[0] if len(vi_pair_all[m_index]) != 1 else 1 for i
                                                                     in vi_pair_all[m_index]]
        end_index += sense_length[m_index]

    unless_count = 0
    for n_index, nws_index in enumerate([i for i in range(len(relation_list_all)) if i not in wsd_index]):
        real_index = n_index - unless_count
        if relation_list_all[nws_index]:
            sim_matrix[end_index:end_index + sense_length_non[real_index], end_index:end_index + sense_length_non[real_index]] = 0
            norm_sim = [i[0] if sense_length_non[real_index] != 1 else 1 for i in vi_pair_all_non[real_index]]
            value_vector[end_index:end_index + sense_length_non[real_index]] = norm_sim
            end_index += sense_length_non[real_index]
        else:
            unless_count += 1

    for index, value in enumerate(value_vector):
        if value[0] == 1:
            sim_matrix[index, :] *= 2
            sim_matrix[:, index] *= 2

    # get all synsets to expand the graph
    count, ccount = 0, 0
    o_sense = list()
    for r_index, word_relation in enumerate(relation_list_all):
        for s_index, sense_relation in enumerate(word_relation):
            if s_index in max_index_all[r_index]:
                dict_pop = list()
                original_sense = sense_relation[0]
                for synset_pair in sense_relation[1].keys():
                    synset_offset = str(wn.synset(synset_pair[0]).offset()).zfill(8)
                    synset_pos = synset_pair[0].split('.')[-2].replace('s', 'a')
                    distance = wn.synset(synset_pair[0]).shortest_path_distance(wn.synset(original_sense))
                    try:
                        if synset_pos in ['n', 'v']:
                            sense_relation[1][synset_pair] = [wn.synset(i).name() for i in
                                                              name['%s_xwn' % synset_pos][synset_offset].split()]
                        else:
                            sense_relation[1][synset_pair] = [wn.synset(i).name() for i in
                                                              name['%s_xwn' % synset_pos][
                                                                  wn.synset(synset_pair[0]).definition()].split()]
                    except:

                        sense_relation[1][synset_pair] = [wn.synsets(i)[0].name() for i in
                                                          sense_relation[1][synset_pair] if wn.synsets(i)]
            else:
                word_relation[s_index] = ('', {})

    graph_nodes, extend_sense = list(), list()
    for r_index, word_relation in enumerate(relation_list_all):
        for s_index, sense_relation in enumerate(word_relation):
            if s_index in max_index_all[r_index]:
                graph_nodes.extend([sense_relation[0]])
                for synset_pair in sense_relation[1].keys():
                    extend_sense.extend([i for i in synset_pair])

    graph_nodes.extend(list(set(' '.join([' '.join([' '.join(sum(d[1].values(), [])) for d in alist]) for alist in
                                          relation_list_all]).split()).difference(set(graph_nodes))))
    graph_nodes.extend(list(set(extend_sense).difference(graph_nodes)))

    graph_matrix = np.zeros((len(graph_nodes), len(graph_nodes)))
    graph_vector = np.ones((len(graph_nodes), 1)) / len(graph_nodes)
    for w_index, word_relation in enumerate(relation_list_all):
        for s_index, sense_relation in enumerate(word_relation):
            if s_index in max_index_all[w_index]:

                for pair in sense_relation[1].keys():
                    wsd_gloss_index = [graph_nodes.index(i) for i in
                                       set(sense_relation[1][pair]).intersection(graph_nodes) if pair[0] == pair[1]]
                    wsd_gloss_index += [
                        graph_nodes.index(pair[1]) if pair[1] in graph_nodes else graph_nodes.index(pair[0])]
                    graph_matrix[graph_nodes.index(pair[0]), wsd_gloss_index] = 1

                    # gloss cluster connection
                    if [j for j in sense_relation[1].keys() if pair[1] == j[0]]:
                        syn_set_index = [graph_nodes.index(i) for i in
                                         set(sense_relation[1][pair]).intersection(graph_nodes)]
                        o_syn_set_index = [graph_nodes.index(i) for i in set(
                            sense_relation[1][[j for j in sense_relation[1].keys() if pair[1] == j[0]][0]]).intersection(graph_nodes)]
                        graph_matrix[
                            [[i] * len(o_syn_set_index) for i in syn_set_index], [o_syn_set_index] * len(syn_set_index)] = 1

    value_vector_all = np.ones((len(graph_nodes), 1)) / len(graph_nodes)
    value_vector_all[:value_vector.shape[0]] = value_vector
    graph_matrix -= np.diag(graph_matrix.diagonal())

    graph_matrix_combine = graph_matrix + graph_matrix.T
    graph_matrix_combine[graph_matrix_combine > 0] = 1
    graph_matrix_combine[:sim_matrix.shape[0], :sim_matrix.shape[0]] = sim_matrix
    graph_matrix_combine -= np.diag(graph_matrix_combine.diagonal())
    helpful_node = [i for i, value in enumerate(graph_matrix_combine.sum(axis=0)) if int(value) != 0]
    reference_vector = graph_matrix_combine.sum(axis=0)[:sim_matrix.shape[0]]
    graph_matrix_combine = graph_matrix_combine[:, helpful_node][helpful_node]
    value_vector_new = value_vector_all[helpful_node]
    graph_matrix_combine /= graph_matrix_combine.sum(axis=0)

    graph_weight_valid = page_rank(graph_matrix_combine, value_vector_new)
    useless_count = 0
    graph_weight = np.zeros(value_vector.shape)
    for index, value in enumerate(reference_vector):
        if value == 0:
            useless_count += 1
            graph_weight[index] = value_vector[index]
        else:
            graph_weight[index] = graph_weight_valid[index-useless_count]

    end_index = 0
    for s_index, wd_index in enumerate(wsd_index):
        sense_weight = list()
        max_sense_index = [i[1] for i in vi_pair_all[s_index]]
        weight_final = graph_weight[end_index:end_index + sense_length[s_index]]
        weight_final /= weight_final.sum()
        end_index += sense_length[s_index]
        for weight in weight_final:
            sense_weight.append(weight)
        weight_all = [float(sense_weight[max_sense_index.index(i)]) if i in max_sense_index else 0 for i in
                      range(len(similarity_list[s_index]))]

        sim_norm = similarity_list[s_index]

        if np.sum(sim_norm) != 0:
            sim_norm = np.array(sim_norm) / np.sum(sim_norm)
        else:
            sim_norm = np.array(sim_norm)

        damping = 0.1
        weight_max = weight_all.index(np.max(weight_all))
        sim_max = np.where(sim_norm == np.max(sim_norm))[0][0]

        similarity_list_path.append(
            np.array(weight_all).reshape((len(weight_all), 1)) * damping + sim_norm * (1 - damping))

    return similarity_list, similarity_list_path


def page_rank(graph_matrix, graph_vector, damping_factor=0.85, iteration=100, delta=0.0001):
    graph_weight_o = np.ones(graph_vector.shape) / graph_vector.shape[0]
    for i in range(iteration):
        graph_weight = damping_factor * np.dot(graph_matrix, graph_weight_o) + (1 - damping_factor) * graph_vector
        if abs(graph_weight - graph_weight_o).sum() < delta:
            break
        else:
            graph_weight_o = graph_weight

    return graph_weight_o


def context_vector(context, data_year):
    """
        the words in the context can be weighted according to their distance to the disambiguated word
        this is not done here
    """
    data_year = data_year
    lsa_matrix, lsa_vocab = retrieve_lsa(data_year)

    topic_num = lsa_matrix.shape[0]

    # the vector representation of the original context
    context_index_o = [lsa_vocab.index(i) for i in context if i in lsa_vocab]
    context_final = [i for i in context if i in lsa_vocab]
    context_mat_o = lsa_matrix[:, context_index_o]

    context_weight_o = np.ones((len(context_final), 1))
    context_vec_o = np.dot(context_mat_o, context_weight_o)
    context_vec = context_vec_o

    return context_vec, topic_num, context


def vector_compress(data_year):
    import numpy as np
    import pickle

    year = data_year
    gloss = list(set([i.strip() for i in open('gloss%s.txt' % year, 'r').readlines()]))
    context = ' '.join([line.strip() for line in open('semeval%s.txt' % year, 'r').readlines()]).split()
    context = [i[0] for i in context]
    words = list(set(gloss + context))

    lsa_ofile, lsa_ofile_vocab = './BNC/eLSA%s' % year, './BNC/eLSA_vocab%s' % year

    def load_lsa(file_name, gloss):
        embedding = dict()
        lsa_matrix = pickle.load(open(file_name[0], 'rb'))
        lsa_vocab = pickle.load(open(file_name[1], 'rb'))
        for word in tqdm(gloss):
            if word in lsa_vocab:
                embedding[word] = lsa_matrix[:, lsa_vocab.index(word)].tolist()
        print('Loaded LSA_mat!')
        return embedding

    embedding = load_lsa((lsa_ofile, lsa_ofile_vocab), words)
    em_words = [i for i in embedding.keys()]
    nonem_words = list(set(words).difference(em_words))
    print(len(nonem_words))

    vocab = [i[0] for i in sorted(embedding.items())]
    embedding_matrix = np.array([i[1] for i in sorted(embedding.items())])
    embedding_matrix = embedding_matrix.astype(float)

    print(embedding_matrix.shape)
    pickle.dump(embedding_matrix.T, open('eLSA%s' % year, 'wb'), -1)
    pickle.dump(vocab, open('eLSA_vocab%s' % year, 'wb'), -1)
