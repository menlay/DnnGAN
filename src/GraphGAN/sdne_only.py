from sdne import *


import sys
sys.path.append("../..")

from collections import Counter


import os
import collections
import tqdm
import multiprocessing
import pickle
import numpy as np
import tensorflow as tf
import config
# import generator
# import discriminator
from src import utils
from src.evaluation import link_prediction as lp
from src.evaluation import ming_test as mt

from graph import *
import time
import sdne_gen
import sdne_dis
import ast


g = Graph()
graph_format = 'edgelist'
print("Reading...")
if graph_format == 'adjlist':
    g.read_adjlist(filename=config.train_filename)
elif graph_format == 'edgelist':
    g.read_edgelist(filename=config.train_filename, weighted=config.weighted,
                    directed=config.directed)
# self.adj = g.G.adj
# self.adj_mat = self.getAdj()
encoder_list = str([g.node_size, config.n_emb])
encoder_layer_list = ast.literal_eval(encoder_list)

# self.root_nodes = list(g.G.nodes)
# self.n_node = len(self.root_nodes)
sdne = SDNE(g, encoder_layer_list=encoder_layer_list,
                          alpha=config.alpha, beta=config.beta, nu1=config.nu1, nu2=config.nu2,
                        learning_rate=config.lr_dis)