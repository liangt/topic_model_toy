#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import numpy as np


def lda(docs_words, num_words, num_topics, max_iter=500, alpha=None, beta=0.01, burn_in=50, lag=20):
    """
    :param docs_words: documents, each represented by the index of each word
    :param num_words: the size of vocabulary
    :param num_topics: the number of topics
    :param max_iter: iteration steps, default is 500
    :param alpha: hyperparameter alpha, default is 50/num_topics
    :param beta: hyperparameter beta, default is 0.01
    :param burn_in: burn_in iterations, default is 50
    :param lag: sampling lag(thinning interval), default is 20
    :return theta: [num_docs, num_topics]
    :return phi: [num_topics, num_words]
    """

    # initialize
    num_docs = len(docs_words)
    num_stats = 0
    if not alpha:
        alpha = 50.0 / num_topics
    docs_topics = np.zeros((num_docs, num_topics))
    topics_words = np.zeros((num_topics, num_words))
    docs_topics_sum = np.zeros(num_docs)
    topics_words_sum = np.zeros(num_topics)
    theta = np.zeros((num_docs, num_topics))
    phi = np.zeros((num_topics, num_words))
    docs_words_topic = []
    for m, doc in enumerate(docs_words):
        temp = []
        for word in doc:
            topic = np.random.randint(num_topics)
            temp.append(topic)
            topics_words[topic][word] += 1
            docs_topics[m][topic] += 1
            topics_words_sum[topic] += 1
        docs_topics_sum[m] = len(doc)
        docs_words_topic.append(temp)

    # Gibbs sampling
    for i in range(max_iter):
        for m, doc in enumerate(docs_words_topic):
            for n, topic in enumerate(doc):
                word = docs_words[m][n]
                topics_words[topic][word] -= 1
                docs_topics[m][topic] -= 1
                topics_words_sum[topic] -= 1
                docs_topics_sum[m] -= 1

                p = np.zeros(num_topics)
                for k in range(num_topics):
                    p[k] = (topics_words[topic][word]+beta)/(topics_words_sum[k]+num_words*beta)*\
                           (docs_topics[m][topic]+alpha)/(docs_topics_sum[m]+num_topics*alpha)
                p /= p.sum()
                new_topic = np.argmax(np.random.multinomial(1, p, 1))

                topics_words[new_topic][word] += 1
                docs_topics[m][new_topic] += 1
                topics_words_sum[new_topic] += 1
                docs_topics_sum[m] += 1

                docs_words_topic[m][n] = new_topic

                if i > burn_in and lag > 0 and i % lag == 0:
                    for d in range(num_docs):
                        for k in range(num_topics):
                            theta[d][k] += (docs_topics[d][k]+alpha)/(docs_topics_sum[d]+num_topics*alpha)

                    for k in range(num_topics):
                        for w in range(num_words):
                            phi[k][w] += (topics_words[k][w]+beta)/(topics_words_sum[k]+num_words*beta)

                    num_stats += 1

    theta /= num_stats
    phi /= num_stats

    return theta, phi


def words_of_topic(dictionary, phi, k):
    """
    :param phi: [num_topics, num_words]
    :param dict_path: dictionary
    :param k: the top k words for each topic
    :return topics: top-k words of each topic, [num_topics, k]
    """

    word_ids = np.argsort(phi, axis=1)
    num_topics, num_words = phi.shape
    if num_words < k:
        k = num_words
    topics = [[] for i in range(num_topics)]
    for i, row in enumerate(word_ids):
        for word in row[:num_words-k-1:-1]:
            topics[i].append(dictionary.get(word))
    return topics

if __name__ == '__main__':
    from cPickle import load
    docs_words = load(open('train_data'))

    from gensim import corpora
    dictionary = corpora.Dictionary.load('dictionary.dict')

    theta, phi = lda(docs_words, len(dictionary), 15)
    topics = words_of_topic(dictionary, phi, 20)
    for i, topic in enumerate(topics):
        print 'Topic %d:' % (i+1)
        print ' '.join(topic), '\n'