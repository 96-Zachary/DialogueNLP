import logging
import torch

import layer


# Hyper Parameters
lower = True
seq_length = 50
report_every = 100000
shuffle = True

def makeVocabulary(filenames, size):
    vocab = layer.Dict([layer.Constants.PAD_WORD, layer.Constants.UNK_WORD,
                          layer.Constants.BOS_WORD, layer.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().replace('\t', ' ').split(' '): # add tab for split
                    if word:
                        vocab.add(word)
    
    vocab.label2idx[''] = 0 # add null str
    originalSize = vocab.size()
    vocab = vocab.prune(size)

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):

    if vocabFile is not None:
        vocab = vocabFile
    
    if vocabFile is None:
    # If a dictionary is still missing, generate it.
        genWordVocab = makeVocabulary(dataFiles, vocabSize)
        vocab = genWordVocab
    
    return vocab

def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, src_INS, src_DEL, tgt = [], [], [], []
    sizes = []
    count, ignored = 0, 0

    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    
    while True:
        src_line = srcF.readline()
        tgt_line = tgtF.readline()
        
        if src_line == '' and tgt_line == '':
            break
        if src_line == '' or tgt_line == '':
            break
        
        src_items = src_line.split('\t')
        
        # 如果src不是一个三元组， 则整个程序停止
        assert len(src_items) == 3
        src_line = src_items[0].strip()
        src_ins = src_items[1].strip()
        src_del = src_items[2].strip()
        tgt_line = tgt_line.strip()
        
        src_words = src_line.split(' ')
        src_ins_words = src_ins.split(' ')
        src_del_words = src_del.split(' ')
        tgt_words = tgt_line.split(' ')
        
        if len(src_words) <= seq_length and len(tgt_words) <= seq_length:
            src += [srcDicts.convert2idx(src_words, layer.Constants.UNK_WORD)]
            src_INS += [tgtDicts.convert2idx(src_ins_words, layer.Constants.UNK_WORD)]
            src_DEL += [tgtDicts.convert2idx(src_del_words, layer.Constants.UNK_WORD)]
            tgt += [tgtDicts.convert2idx(tgt_words,
                                         layer.Constants.UNK_WORD,
                                         layer.Constants.BOS_WORD,
                                         layer.Constants.EOS_WORD)]
            sizes += [len(src_words)]
        else:
            ignored += 1
        count += 1
    srcF.close()
    tgtF.close()
    
    if shuffle:
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_INS = [src_INS[idx] for idx in perm]
        src_DEL = [src_DEL[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    src_INS = [src_INS[idx] for idx in perm]
    src_DEL = [src_DEL[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    return src, src_INS, src_DEL, tgt

def prepare_data_online(train_src, vocab_src, train_tgt, vocab_tgt):
    # 准备标签和数据映射字典
    size = 30000
    dicts = {}

    dicts['src'] = initVocabulary('source', [train_src], vocab_src, size)
    dicts['tgt'] = initVocabulary('target', [train_tgt], vocab_tgt, size)

    # 准备训练数据集
    train = {}
    train['src'], train['ins'], train['del'], train['tgt'] = makeData(train_src,
                                                                      train_tgt,
                                                                      dicts['src'],
                                                                      dicts['tgt'])
    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
