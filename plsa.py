#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

from gensim import corpora
import numpy as np


def load_data(path):
    """
    :param path: corpus path
    :return terms_docs: words-documents matrix
    """

    # train_data: documents set, each document is a collection of (word_id, frequent)
    train_data = corpora.MmCorpus(path)
    num_terms = train_data.num_terms
    num_docs = train_data.num_docs
    terms_docs = np.zeros((num_terms, num_docs))

    for i, d in enumerate(train_data):
        for t_f in d:
            terms_docs[t_f[0], i] = t_f[1]
    return terms_docs


def plsa(terms_docs, num_topics, max_iter=100):
    """
    :param terms_docs: term-document matrix, [num_words, num_docs]
    :param num_topics: the number of topics
    :param max_iter: iteration steps, default is 100
    :return p_w_z: p(w|z), [num_terms, num_topics]
    :return p_z_d: p(z|d), [num_topics, num_docs]
    :return log_l: log-likelihood
    """

    # initialize
    num_terms, num_docs = terms_docs.shape
    p_w_z = np.random.rand(num_terms, num_topics)
    p_w_z /= p_w_z.sum(axis=1).reshape((-1, 1))
    p_z_d = np.random.rand(num_topics, num_docs)
    p_z_d /= p_z_d.sum(axis=1).reshape((-1, 1))
    p_w_d = np.zeros((num_terms, num_docs))
    p_z_w_d = np.zeros((num_topics, num_terms, num_docs))
    log_l = np.zeros(max_iter)

    # EM
    for i in range(max_iter):
        # E step
        # p(z|d, w)
        for w in range(num_terms):
            for d in range(num_docs):
                p_w_d[w][d] = 0
                for z in range(num_topics):
                    p_z_w_d[z, w, d] = p_w_z[w, z] * p_z_d[z, d]
                    p_w_d[w][d] += p_z_w_d[z, w, d]
                p_z_w_d[:, w, d] /= p_w_d[w][d]
        log_l[i] = np.sum(terms_docs * np.log(p_w_d))
        print 'Iteration %d, log-likelihood is %f' % (i+1, log_l[i])

        # M step
        # p(w|z)
        for z in range(num_topics):
            p_w_z[:, z] = np.sum(terms_docs * p_z_w_d[z], axis=1)
        p_w_z /= p_w_z.sum(axis=1).reshape((-1, 1))

        # p(z|d)
        for z in range(num_topics):
            p_z_d[z, :] = np.sum(terms_docs * p_z_w_d[z], axis=0)
        p_z_d /= p_z_d.sum(axis=1).reshape((-1, 1))

    return p_w_z, p_z_d, log_l


def words_of_topic(dict_path, p_w_z, k):
    """
    :param p_w_z: p(w|z), [num_terms, num_topics]
    :param dict_path: dictionary path
    :param k: the top k words for each topic
    :return topics: top-k words of each topic, [num_topics, k]
    """
    dictionary = corpora.Dictionary.load(dict_path)

    word_ids = np.argsort(p_w_z, axis=0)
    num_words, num_topics = p_w_z.shape
    if num_words < k:
        k = num_words
    topics = [[] for i in range(num_topics)]
    for row in word_ids[:num_words-k-1:-1]:
        for i, col in enumerate(row):
            topics[i].append(dictionary.get(col))
    return topics

if __name__ == '__main__':
    terms_docs = load_data('corpus.mm')
    p_w_z, p_z_d, log_l = plsa(terms_docs, 15, 100)

    topics = words_of_topic('dictionary.dict', p_w_z, 20)
    for i, topic in enumerate(topics):
        print 'Topic %d:' % (i+1)
        print ' '.join(topic), '\n'

    import matplotlib.pyplot as plt
    plt.figure(1)
    x = range(log_l.shape[0])
    plt.plot(x, log_l)
    plt.show()