# -*- coding:utf8 -*-
"""
Creared on 2018/04/07
Author: fangdenghui
Functuon: chinese word find relate of class algorithm
Reference: https://github.com/xiulonghan/wordSeg
"""

import newWordsFind
import argparse

def save_list(filepath, mylist):
    '''
    Desc 保存list列表到文件中
    '''

    fw = open(filepath, 'w')

    for line in mylist:
        fw.write(line+'\n')
    fw.close()

def find_words_of_class(input_pos, input_oth, rate=10, max_word_length=5, min_tf=0.000005,
        min_infor_ent=1.0, min_pmi=6.0, output=None,
        tmp_pos_words='data/tmp_pos_words.csv',
        tmp_oth_words='data/tmp_oth_words.csv'):
    '''
    Desc 基于文件词汇分布，找出pos类别相关的词汇

    Args:
        input_pos 新词发现相关类的文本路径
        input_oth 对比文本
        rate 对比指标倍率，当tf, infor_ent, pmi正负相差rate倍，认定和正样本相关性较大
        max_word_length 新词发现时最低词长度
        min_tf 最低词频要求
        min_infor_ent 最低自由度
        min_pmi 最低凝聚度
        output 保存正例相关词汇
        tmp_pos_words 存储正例新词词汇信息
        tmp_oth_words 存储负例新词词汇信息
    '''
    print 'args:'
    print 'input_pos:', input_pos
    print 'input_oth:', input_oth
    print 'rate:', rate
    print 'max_word_length:', max_word_length
    print 'min_tf:', min_tf
    print 'min_infor_ent:', min_infor_ent
    print 'min_pmi:', min_pmi
    print 'output:', output
    doc_pos = open(input_pos).read()
    doc_oth = open(input_oth).read()

    words_pos = newWordsFind.SegDocument(doc_pos)
    words_oth = newWordsFind.SegDocument(doc_oth)

    oth_dic = dict()

    for item in words_oth.word_tf_pmi_ent:
        oth_dic[item[0]] = item

    pos_list = list()

    for item_pos in words_pos.word_tf_pmi_ent:
        if oth_dic.has_key(item_pos[0]):
            item_oth = oth_dic[item_pos[0]]
            if item_pos[1] / item_oth[1] >=rate:
                pos_list.append(item_pos[0])
        else:
            pos_list.append(item_pos[0])

    save_list(output, pos_list)

if __name__ == '__main__':
    # 添加两种类别的文件路径，根据不同分布区分新词

    parser = argparse.ArgumentParser('argument for find new words relate with class')
    parser.add_argument('-input_pos', default='data/news_game.data', type=str, help='input pos file')
    parser.add_argument('-input_oth', default='data/news_other.data', type=str, help='input other file')
    parser.add_argument('-output', default='data/game_keywords.txt', type=str, help='output keywords file')

    parser.add_argument('-rate', default=10, type=int, help='occur rate of pos vs other')
    parser.add_argument('-max_word_length', default=5, type=int)
    parser.add_argument('-min_tf', default=0.000005, type=float)
    parser.add_argument('-min_infor_ent', default=1.0, type=float)
    parser.add_argument('-min_pmi', default=6.0, type=float)

    args = parser.parse_args()

    find_words_of_class(args.input_pos, args.input_oth, rate=args.rate, max_word_length=args.max_word_length,
            min_tf=args.min_tf, min_infor_ent=args.min_infor_ent, min_pmi=args.min_pmi, output=args.output)
