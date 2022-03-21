import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

# Create a simple random network with K nodes a sparsity level of p
# Each event induces impulse responses of length dt_max on connected nodes
K = 3
p = 0.25
dt_max = 20
network_hypers = {"p": p, "allow_self_connections": False}
true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
    K=K, dt_max=dt_max, network_hypers=network_hypers)

# Generate T time bins of events from the the model
# S is the TxK event count matrix, R is the TxK rate matrix
S,R = true_model.generate(T=100)
true_model.plot()