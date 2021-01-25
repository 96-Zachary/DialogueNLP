from __future__ import division


import torch
import numpy as np

class Beam(object):
    def __init__(self, generator, size, batch_first=True):
        self.batch_first = batch_first
        self.generator = generator
        self.beam_size = size

    def search(self, g_outputs, targets, wether_gene=False):
        if wether_gene:
            pred_gene = g_outputs
        else:
            '''
            g_outputs = [seq_len, batch_size, hidden_dim]
            '''
            # pred_gene = [seq_len, batch_size, voc_dim]
            pred_gene = self.generator(g_outputs.view(-1, g_outputs.shape[-1])).view(g_outputs.shape[0],g_outputs.shape[1], -1)
        if self.batch_first:
            pred_gene = pred_gene.permute(1, 0, 2)
            tgt = targets.permute(1, 0)
            seq_len = tgt.shape[0]
        else:
            tgt = targets
            seq_len = tgt.shape[0]
        w = np.random.randint(seq_len - 1, size=(tgt.shape[1], int(0.15*self.beam_size*seq_len)))

        # pred = []
        for j in range(pred_gene.shape[1]):
            # gene_sents = [seq_len, voc_dim]
            gene_sents = pred_gene[:, j, :]
            pred_record, pred_score = [], []
            # for all word in sentences
            for word in gene_sents:
                sorts, sort_idx = torch.sort(word)
                tmp_records = []
                tmp_scores = []
                # for beam search size record
                for i in range(self.beam_size):
                    tmp_pred = sort_idx[-(i + 1)]
                    score = sorts[i]
                    tmp_records.append([tmp_pred.item()])
                    tmp_scores.append(score)
                # print(tmp_records)
                if len(pred_record) == 0:
                    pred_record = tmp_records
                    pred_score = tmp_scores
                else:
                    pred_records = []
                    pred_scores = []
                    for i in range(len(pred_record)):
                        exiting = pred_record[i]
                        score = pred_score[i]
                        for tmp in tmp_records:
                            pred_records.append(exiting + tmp)
                        for tmp in tmp_scores:
                            pred_scores.append((score + tmp).item())
                    socre_sort, score_sort_idx = torch.sort(torch.tensor(pred_scores), descending=True)

                    pred_record = []
                    pred_score = []
                    for idx in score_sort_idx[:self.beam_size]:
                        pred_record.append(pred_records[idx.item()])
                        pred_score.append(pred_scores[idx.item()])
            if j == 0:
                pred = torch.tensor(pred_record[0]).unsqueeze(1)
            else:
                pred = torch.cat([pred, torch.tensor(pred_record[0]).unsqueeze(1)], dim=1)

        j = 0
        for idx in w:
            pred[idx, j] = tgt[idx, j]
            j += 1
        return pred