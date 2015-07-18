#!/usr/bin/env python
# coding: utf-8 

__author__ = 'liangliang'

import os, nltk
import numpy as np
from cPickle import dump
from gensim import corpora


def handle(data_path, stopword_path):
    stopword = open(stopword_path).read().split()
    docs = []
    for d in os.listdir(data_path):
        for f in os.listdir(data_path + '/' + d):
            txt = open(data_path + '/' + d + '/' + f).read().decode("utf-8")

            # fen ci
            words = [w.lower() for w in nltk.word_tokenize(txt) if w.isalpha() and w.lower() not in stopword]

            # ti qu ci gan
            words = [nltk.PorterStemmer().stem(w) for w in words]

            docs.append(words)

    print >>open('plain_text', 'w'), docs

    # generate dictionary
    dictionary = corpora.Dictionary(docs)
    dictionary.save('dictionary.dict')

    # translate document to vector of words frequency
    corpus = [dictionary.doc2bow(text) for text in docs]
    corpora.MmCorpus.serialize('corpus.mm', corpus)

    doc = [[] for i in docs]
    for i, d in enumerate(docs):
        for w in d:
            doc[i].append(dictionary.token2id[w])
    dump(doc, open('train_data', 'w'))

if __name__ == '__main__':
    handle('data', 'stopword.txt')