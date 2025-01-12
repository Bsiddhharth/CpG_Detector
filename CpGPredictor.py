import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from functools import partial
from typing import Sequence

# LSTM_HIDDEN = 128
# LSTM_LAYER = 1
# batch_size = 64
# learning_rate = 0.001
# epoch_num = 20

class CpGPredictor(nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(5, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        l_out, _ = self.lstm(x)
        output = l_out[:, -1, :]
        logits = self.classifier(output)
        return logits


def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}  #
int2dna = { i: a for a, i in zip(alphabet, range(5))}  #

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)


def prepare_data(num_samples=100):
    
    X_dna_seqs_train = list(rand_sequence(num_samples))
    
    temp = [list(intseq_to_dnaseq(x))  for x in X_dna_seqs_train] 

    y_dna_seqs = [count_cpgs(seq) for seq in list(map(''.join, temp))]

    return np.array(X_dna_seqs_train), np.array(y_dna_seqs)
