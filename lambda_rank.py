import torch
from torch import nn
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from torch.functional import F

datum = namedtuple("datum", "relevance features")

path_train = "MSLR/Fold1/train.txt"
path_test = "MSLR/Fold1/test.txt"
small_path = "MSLR/Fold1/sample_text.txt"

# todo : normalize data


class NN(nn.Module):
    def __init__(self, n1=64, n2=64):
        super().__init__()
        self.h1 = nn.Linear(136, n1)
        self.h2 = nn.Linear(n1, n2)
        self.out = nn.Linear(n2, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)

        x = self.out(x)
        x = torch.sigmoid(x)

        return x


def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            split_line = line.split(":")
            feat = torch.tensor([eval(frag.split()[0]) for frag in split_line[2:]])
            data.append(datum(relevance=eval(split_line[0][0]), features=feat))

    return data


class RankNet:
    def __init__(self):
        self.nn = NN()

    def fit(self, learning_rate=0.1, epochs=100):
        pass


"""
compute lambda: lambda_i,j = sigmoid(si-sj) * delta_NDCG
"""


def compute_single_ndcgs(scores):
    nb_scores = len(scores)
    num = np.power(2, scores) - 1
    denom = 1 / np.log2(np.arange(2, nb_scores + 2))

    return np.multiply(num.reshape(nb_scores, 1), denom.reshape(1, nb_scores))


class LambdaRank:
    def __init__(self):
        pass

    def compute_lambda(self, true_scores, predicted_scores, ordered_pairs):
        nb_scores = len(true_scores)
        singles_dcgs = compute_single_ndcgs(true_scores)
        lambdas = np.zeros(nb_scores)
        for (i, j) in ordered_pairs:
            delta_ndcg = abs(singles_dcgs[i, j] + singles_dcgs[j, i] - singles_dcgs[i, i] - singles_dcgs[j, j])
            ds = torch.sigmoid(- (predicted_scores[i] - predicted_scores[j])) * delta_ndcg
            lambdas[i] += ds
            lambdas[j] -= ds

        return lambdas

    def compute_ordered_pairs(self, scores):
        op = []
        nb_scores = len(scores)
        for i in range(nb_scores):
            for j in range(nb_scores):
                if scores[i] > scores[j]:
                    op.append((i, j))

        return op

    def fit(self, epochs=100):
        for k in tqdm(range(epochs)):
            # query by query
            scores = []
            true_scores = []
            for datum in data_train:
                scores.append(nn(datum.feature))
                true_scores.append(datum.relevance)

            op = self.compute_ordered_pairs(scores)
            lambdas = self.compute_lambda(true_scores, scores, op)

            # update net with stochastic gradient w = w + learning_rate * lambda * ds/dw


if __name__ == "__main__":
    data_train = load_data(small_path)
    
