# -*- coding:utf-8 -*-
"""
Chinese word segmentation algorithm without corpus.
Author:aluka.han
Email:aluka_hxl@gmail.com
Reference:
    https://github.com/Moonshile/ChineseWordSegmentation
    http://www.matrix67.com/blog/archives/5044
    https://zlc1994.com/2017/01/04/
"""
import os
import re
import time
import math
import pandas as pd
import sys
from leftRightEntropy import cal_infor_entropy
from createCandidateWords import extract_cand_words, gen_bigram
from pyutil import pyredis

class GetWordInfo(object):
    """
    Store information of each word, including it's frequency, left neighbors and right neighbors
    """
    def __init__(self, text):
        """
        init function,the text is the word.
        :param text:the string will be compute,include fre,PMI,information entropy.
        """
        super(GetWordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = dict()
        self.right = dict()
        self.pmi = 0

    def update_att(self, left, right):
        """
        Increase frequency of this word, then append left/right neighbors.
        :param left: left neighbor set
        :param right: right neighbor set
        """
        self.freq += 1
        if left:
            self.left[left] = self.left.get(left, 0) + 1
        if right:
            self.right[right] = self.right.get(right, 0) + 1

    def compute_indexes(self, length):
        """
        Based on the update_att,compute tf and entropy of this word
        :param length: the length of document.
        """
        self.freq /= length
        self.left = cal_infor_entropy(self.left)
        self.right = cal_infor_entropy(self.right)

    def compute_info_entropy(self, words_dict):
        """
        compute the text's PMI, and select the min PMI for all bi-gram.
        :param words_dict: it's contain all candidate word information
        """
        sub_parts = gen_bigram(self.text)
        if len(sub_parts) > 0:
            self.pmi = min(map(
                lambda (left_g, right_g): math.log(
                    self.freq/(words_dict[left_g].freq*words_dict[right_g].freq)), sub_parts))


class SegDocument(object):
    """
    Main class for Chinese word segmentation
    1. Generate words from a long enough document
    2. Do the segmentation work with the document
    """
    def __init__(self, doc, max_word_len=5, min_tf=0.000005, min_infor_ent=1.0, min_pmi=5):
        super(SegDocument, self).__init__()
        self.max_word_len = max_word_len
        self.min_tf = min_tf
        self.min_info_ent = min_infor_ent
        self.min_pmi = min_pmi
        self.word_infos = self.gen_words(doc)
        # calculate the average value for every index.
        word_count = float(len(self.word_infos))
        self.avg_len = sum(map(lambda w: len(w.text), self.word_infos)) / word_count
        self.avg_freq = sum(map(lambda w: w.freq, self.word_infos)) / word_count
        self.avg_left_entropy = sum(map(lambda w: w.left, self.word_infos)) / word_count
        self.avg_right_entropy = sum(map(lambda w: w.right, self.word_infos)) / word_count
        self.avg_pmi = sum(map(lambda w: w.pmi, self.word_infos)) / word_count
        self.avg_info_ent = sum(map(lambda w: min(w.left, w.right), self.word_infos)) / word_count
        # Filter out the results satisfy all the requirements
        filter_function = lambda v: len(v.text) > 1 and v.pmi > self.min_pmi and\
                    v.freq > self.min_tf and min(v.left, v.right) > self.min_info_ent
        self.word_tf_pmi_ent = map(lambda w: (w.text, w.freq, w.pmi, min(w.left, w.right)),
                                   filter(filter_function, self.word_infos))

    def gen_words(self, doc):
        """
        Generate all candidate words with their frequency/pmi/infor_entropy
        :param doc:the document used for words generation
        :return:
        """
        doc = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#”“￥：%……&*（）]+".decode("utf8"),
                      "".decode("utf8"), doc.decode('utf8'))
        word_cands = dict()
        # iterate all candidate of words
        doc_lens = len(doc)
        cnt = 0
        for i in xrange(doc_lens):
            for j in xrange(i + 1, min(i + 1 + self.max_word_len, doc_lens + 1)):
                word = doc[i:j]
                cnt += 1
                if cnt % 10000 == 0:
                    print 'process word:', cnt
                    sys.stdout.flush()
                if word not in word_cands:
                    word_cands[word] = GetWordInfo(word)
                word_cands[word].update_att(doc[i-1:i], doc[j:j+1])
        del doc

        print 'compute tf and info'
        # compute the tf and info_entropy
        for word in word_cands:
            word_cands[word].compute_indexes(doc_lens)

        #print 'sort'
        # compute PMI for every word, if len(word)>1
        #values = sorted(word_cands.values(), key=lambda x: len(x.text))

        values = word_cands.values()
        print 'cal pmi'
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_info_entropy(word_cands)
        print 'ret sort'
        return sorted(values, key=lambda v: v.freq, reverse=True)[:1000000]


if __name__ == '__main__':

    start = time.clock()
    path = os.path.abspath('.')
    #doc = open(path + '\\zibenlun.txt', 'r').read()
    doc = open(path + '/news_game.txt', 'r').read()
    xx = SegDocument(doc)
    print '西游记平均长度:', xx.avg_len
    print '西游记平均词频:', xx.avg_freq
    print '西游记平均PMI:', xx.avg_pmi
    print '西游记平均信息熵:', xx.avg_info_ent
    word = []
    for item in xx.word_tf_pmi_ent:
        word.append([item[0], item[1], item[2], item[3]])
    df = pd.DataFrame(word, columns=['word', 'tf', 'pmi', 'info_ent'])
    df.to_csv(path+'/newsgameSeg.csv', index=False)
    end = time.clock()
    print end-start
