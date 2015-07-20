#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import numpy as np


class GibbsLDA:
    """
    Collapsed LDA: implemented by Gibbs sampling.
    """

    def __init__(self, docs_words, num_words, num_topics, max_iter=1000, alpha=None, beta=0.01, burn_in=800, lag=20):
        self.docs_words = docs_words     # documents, each represented by the index of each word
        self.num_docs = len(docs_words)  # the number of documents
        self.num_words = num_words       # the size of vocabulary
        self.num_topics = num_topics     # the number of topics
        self.max_iter = max_iter         # iteration steps, default is 1000
        self.beta = beta                 # hyper_parameter beta, default is 0.01
        self.burn_in = burn_in           # burn_in iterations, default is 800
        self.lag = lag                   # sampling lag(thinning interval), default is 20
        if alpha:                        # hyper_parameter alpha, default is 50/num_topics
            self.alpha = alpha
        else:
            self.alpha = 50.0 / num_topics
        self.theta = np.zeros((self.num_docs, self.num_topics))
        self.phi = np.zeros((self.num_topics, self.num_words))

        # the following two variables are needed in inference
        self.topics_words = np.zeros((num_topics, num_words))
        self.topics_words_sum = np.zeros(num_topics)

    def fit(self):
        docs_topics = np.zeros((self.num_docs, self.num_topics))
        docs_topics_sum = np.zeros(self.num_docs)
        num_stats = 0   # updating times of theta and phi
        docs_words_topic = []
        for m, doc in enumerate(docs_words):
            temp = []
            for word in doc:
                init_topic = np.random.randint(self.num_topics)
                temp.append(init_topic)
                self.topics_words[init_topic][word] += 1
                docs_topics[m][init_topic] += 1
                self.topics_words_sum[init_topic] += 1
            docs_topics_sum[m] = len(doc)
            docs_words_topic.append(temp)

        # Gibbs sampling
        for ite in range(self.max_iter):
            print 'Iteration %d' % ite
            for m, doc in enumerate(docs_words_topic):
                for n, topic in enumerate(doc):
                    word = docs_words[m][n]
                    self.topics_words[topic][word] -= 1
                    docs_topics[m][topic] -= 1
                    self.topics_words_sum[topic] -= 1
                    docs_topics_sum[m] -= 1

                    p = (self.topics_words[:, word]+self.beta)/(self.topics_words_sum+self.num_words*self.beta)*\
                        (docs_topics[m, :]+self.alpha)/(docs_topics_sum[m]+self.num_topics*self.alpha)
                    p /= p.sum()
                    new_topic = np.argmax(np.random.multinomial(1, p, 1))

                    self.topics_words[new_topic][word] += 1
                    docs_topics[m][new_topic] += 1
                    self.topics_words_sum[new_topic] += 1
                    docs_topics_sum[m] += 1

                    docs_words_topic[m][n] = new_topic

            if ite > self.burn_in and self.lag > 0 and ite % self.lag == 0:
                self.theta += (docs_topics + self.alpha) / (docs_topics_sum[:, np.newaxis] + self.num_topics * self.alpha)
                self.phi += (self.topics_words + self.beta) / (self.topics_words_sum[:, np.newaxis] + self.num_words * self.beta)
                num_stats += 1

        self.theta /= num_stats
        self.phi /= num_stats

    def predict(self, docs_words):
        if len(docs_words.shape) == 1:
            docs_words = docs_words[np.newaxis, :]
        num_docs = len(docs_words)
        docs_topics = np.zeros((num_docs, self.num_topics))
        topics_words = np.zeros((self.num_topics, self.num_words))
        docs_topics_sum = np.zeros(num_docs)
        topics_words_sum = np.zeros(self.num_topics)
        docs_topics = np.zeros((num_docs, self.num_topics))
        docs_topics_sum = np.zeros(num_docs)
        theta = np.zeros((num_docs, self.num_topics))
        num_stats = 0   # updating times of theta and phi
        docs_words_topic = []
        for m, doc in enumerate(docs_words):
            temp = []
            for word in doc:
                init_topic = np.random.randint(self.num_topics)
                temp.append(init_topic)
                topics_words[init_topic][word] += 1
                docs_topics[m][init_topic] += 1
                topics_words_sum[init_topic] += 1
            docs_topics_sum[m] = len(doc)
            docs_words_topic.append(temp)

        # Gibbs sampling
        for ite in range(self.max_iter):
            print 'Iteration %d' % ite
            for m, doc in enumerate(docs_words_topic):
                for n, topic in enumerate(doc):
                    word = docs_words[m][n]
                    topics_words[topic][word] -= 1
                    docs_topics[m][topic] -= 1
                    topics_words_sum[topic] -= 1
                    docs_topics_sum[m] -= 1

                    p = (self.topics_words[:, word] + topics_words[:, word] + self.beta)/\
                        (self.topics_words_sum + topics_words_sum + self.num_words*self.beta)*\
                        (docs_topics[m, :]+self.alpha)/(docs_topics_sum[m]+self.num_topics*self.alpha)
                    p /= p.sum()
                    new_topic = np.argmax(np.random.multinomial(1, p, 1))

                    topics_words[new_topic][word] += 1
                    docs_topics[m][new_topic] += 1
                    topics_words_sum[new_topic] += 1
                    docs_topics_sum[m] += 1

                    docs_words_topic[m][n] = new_topic

            if ite > self.burn_in and self.lag > 0 and ite % self.lag == 0:
                self.theta += (docs_topics + self.alpha) / (docs_topics_sum[:, np.newaxis] + self.num_topics * self.alpha)
                num_stats += 1

        theta /= num_stats
        return theta


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

    lda = GibbsLDA(docs_words, len(dictionary), 15)
    lda.fit()
    topics = words_of_topic(dictionary, lda.phi, 20)
    for i, topic in enumerate(topics):
        print 'Topic %d:' % (i+1)
        print ' '.join(topic), '\n'