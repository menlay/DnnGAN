import os
import sys

sys.path.append("/home/ming/gittest/GraphGAN_数据半采样半SDNE/")
# sys.path.append("/home/ming/gittest/GraphGAN_数据半采样/src/GraphGAN/")
os.chdir('/home/ming/gittest/GraphGAN_数据半采样半SDNE/src/GraphGAN/')

from collections import Counter
import collections
import tqdm
import multiprocessing
import pickle
import numpy as np
import tensorflow as tf
import config
import generator
# import discriminator
from src import utils
from src.evaluation import link_prediction as lp
# from src.evaluation import ming_test as mt
from src.evaluation.evalne.utils import preprocess as pp
from evalne.evaluation.evaluator import LPEvaluator as LP
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.evaluator import NREvaluator
from evalne.evaluation.evaluator import NCEvaluator
from src.evaluation.evalne.evaluation.score import Scoresheet
sys.path.append('/home/ming/gittest/EvalNE/evalne/evaluation/')
from split import EvalSplit


from graph import *
import time
# import sdne_gen
import sdne_dis1_2 as sdne_dis
import ast
# from sklearn.externals import joblib

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import networkx as nx

import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
train_file = '/home/ming/gittest/EvalNE/evalne/tests/data/cora/cora_edgelist.txt'


G1 = pp.load_graph(train_file, directed=True)
G, _ = pp.prep_graph(G1, relabel=True, del_self_loops=False)
labels_ = pp.read_labels('/home/ming/gittest/EvalNE/evalne/tests/data/cora/cora_labels.txt', idx_mapping=_)
traintest_split = EvalSplit()
traintest_split.compute_splits(G, train_frac=0.6, owa=False)
# nre = NR(traintest_split, dim=128)
# nlp = LP(traintest_split, dim=128)
nce = NCEvaluator(G, labels_, 'blogCatalog', num_shuffles=2, traintest_fracs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                  trainvalid_frac=0.6, dim=128)

print("start training...")
X_gen = pp.read_node_embeddings('/home/ming/gittest/EvalNE/result4(分类train_frac)/GraphGAN_network_edgelist_dis_.emb', traintest_split.TG.nodes, 128, ' ', 'MY_model')
# results_gen = nc.evaluate_ne(data_split=traintest_split, X=X_gen, method='MY_model', edge_embed_method='hadamard')

results_gen = nce.evaluate_ne(X_gen, method_name='DnnGAN')
print(results_gen)