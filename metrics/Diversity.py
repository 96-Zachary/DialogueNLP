import torch
import torch.nn as nn
import layer


def distinct_1(preds, targets):
    '''
    Relevance Average
    preds = [len, batch_size]
    targets = [len, batch_size]
    embed is the embedding layer in model
    '''
    lens, nums = preds.shape

    dis = []
    for j in range(nums):
        correct, error = 0, 0
        true_idx = torch.where(targets[:, j] != layer.Constants.PAD)[0]
        tmp_pred = preds[true_idx, j]
        tmp_tgt = targets[true_idx, j]

        for word in tmp_pred:
            if word in tmp_tgt:
                correct += 1
            else:
                error += 1
        dis.append(error / lens)
    dis_1 = sum(dis)

    return dis_1, nums


def distinct_2(preds, targets):
    '''
    Relevance Average
    preds = [len, batch_size]
    targets = [len, batch_size]
    embed is the embedding layer in model
    '''
    def bigrams(lists):
        l = [[lists[i], lists[i + 1]] for i in range(len(lists) - 1)]
        return l

    lens, nums = preds.shape

    dis = []
    for j in range(nums):
        correct, error = 0, 0
        true_idx = torch.where(targets[:, j] != layer.Constants.PAD)[0]
        tmp_pred = bigrams(preds[true_idx, j])
        tmp_tgt = bigrams(targets[true_idx, j])

        for word in tmp_pred:
            if word in tmp_tgt:
                correct += 1
            else:
                error += 1
        dis.append(error / lens)
    dis_2 = sum(dis)

    return dis_2, nums