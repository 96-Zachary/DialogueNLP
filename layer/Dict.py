import torch
import pickle

class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idx2label = {}
        self.label2idx = {}
        self.freq = {}
        self.lower = lower
        
        self.special = []
        
        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)
        
    def size(self):
        return len(self.idx2label)
    
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)
    
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]
    
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        # update in both idx2label & label2idx
        if idx is not None:
            self.idx2label[idx] = label
            self.label2idx[label] = idx
        else:
            if label in self.label2idx:
                idx = self.label2idx[label]
            else:
                idx = len(self.idx2label)
                self.idx2label[idx] = label
                self.label2idx[label]= idx
        
        if idx not in self.freq:
            self.freq[idx] = 1
        else:
            self.freq[idx] += 1
        
        return idx
    
    # 重置字典的大小
    def prune(self, size):
        if size >= self.size():
            return self
        freq = torch.Tensor(
                [self.freq[i] for i in range(len(self.freq))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower

        for i in self.special:
            newDict.addSpecial(self.idx2label[i])

        for i in idx[:size]:
            newDict.add(self.idx2label[int(i)])

        return newDict
    
    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label2idx[key]
        except KeyError:
            return default
            
    def getLabel(self, idx, default=None):
        try:
            return self.idx2label[idx]
        except KeyError:
            return default

    def convert2idx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)
        
    def convert2Labels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels


def save_dict(dicts, name, filename):
    if name == None:
        rw = dicts
    else:
        rw = dicts[name]
    output_hal = open(filename, 'wb')
    str = pickle.dumps(rw)
    output_hal.write(str)
    output_hal.close()


def load_dict(filename):
    with open(filename, 'rb') as file:
        x = pickle.loads(file.read())

    return x

