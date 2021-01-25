import torch
import torch.nn as nn

def r_avg(preds, targets, embed):
    '''
    Relevance Average
    preds = [len, batch_size]
    targets = [len, batch_size]
    embed is the embedding layer in model
    '''
    pred_emb = embed(preds)
    tgt_emb = embed(targets)

    pred_avg_emb = torch.mean(pred_emb, dim=0)
    tgt_avg_emb = torch.mean(tgt_emb, dim=0)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(pred_avg_emb, tgt_avg_emb)

    sums = sum(similarity.detach().cpu().numpy())
    nums = similarity.shape[0]

    return sums, nums


def r_greedy(preds, targets, embed):
    import numpy as np
    '''
    Relevance Greedy
    preds = [len, batch_size]
    targets = [len, batch_size]
    embed is the embedding layer in model
    '''
    def g(q1, q2, length):
        # q1==q2 = [len, batch_size]
        sums = torch.zeros(q1.shape[1])
        for word in q1:
            word = word.unsqueeze(0).repeat(length, 1, 1)

            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            sim = torch.max(cos(word, q2), dim=0)[0].detach().cpu().numpy()
            sums += sim
        return sums/length

    # [len, batch_size, embed_size]
    pred_emb = embed(preds)
    tgt_emb = embed(targets)
    length = preds.shape[0]

    g1 = g(pred_emb, tgt_emb, length).detach().cpu().numpy()
    g2 = g(tgt_emb, pred_emb, length).detach().cpu().numpy()
    sums = sum((g1 + g2) / 2)

    return sums, preds.shape[1]


def r_extrema(preds, targets, embed):
    '''
    Relevance Extrema
    preds = [len, batch_size]
    targets = [len, batch_size]
    embed is the embedding layer in model
    '''
    pred_emb = embed(preds)
    tgt_emb = embed(targets)

    pred_min_emb = torch.min(pred_emb, dim=0)[0]
    tgt_min_emb = torch.min(tgt_emb, dim=0)[0]
    pred_absmin_emb = abs(pred_min_emb)
    tgt_absmin_emb = abs(tgt_min_emb)

    pred_max_emb = torch.max(abs(pred_emb), dim=0)[0]
    tgt_max_emb = torch.max(abs(tgt_emb), dim=0)[0]

    pred_extrema = torch.where(pred_max_emb > pred_absmin_emb, pred_max_emb, pred_min_emb)
    tgt_extrema = torch.where(tgt_max_emb > tgt_absmin_emb, tgt_max_emb, tgt_min_emb)

    #     print(pred_max_emb[0, :10], tgt_max_emb[0, :10])
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(pred_extrema, tgt_extrema)

    sums = sum(similarity.detach().cpu().numpy())
    nums = similarity.shape[0]

    return sums, nums