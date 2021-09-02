# 增加个epoch生成器和鉴别器的测评

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
from src.evaluation.evalne.evaluation.split import EvalSplit

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


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default=config.train_filename,
                        help='Input graph file')
    parser.add_argument('--output', default=config.emb_filenames[0],
                        help='Output representation file')
    parser.add_argument('--representation-size', default=config.n_emb, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=config.n_epochs, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--epochs-gen', default=config.n_epochs_gen, type=int,
                        help='The number of iterations of the generator in each loop')
    parser.add_argument('--epochs-dis', default=config.n_epochs_dis, type=int,
                        help='The number of iterations of the generator in each loop')
    parser.add_argument('--cache-filename', default=config.cache_filename,
                        help='Tree structure cache file')
    parser.add_argument('--pretrain-filename', default=config.pretrain_emb_filename_g,
                        help='Tree structure cache file')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')

    args = parser.parse_args()
    return args


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


class GraphGAN(object):
    def __init__(self, args):
        # 命令行传参,到EvalNE进行评估，output_file可以进行选择
        self.input_file = args.input
        self.output_file = args.output
        self.n_emb = args.representation_size
        self.n_epoch = args.epochs
        self.n_epoch_gen = args.epochs_gen
        self.n_epoch_dis = args.epochs_dis
        self.cache_filename = args.cache_filename
        config.directed = args.directed

        t1 = time.time()
        g = Graph()
        graph_format = 'edgelist'
        print("Reading...")
        if graph_format == 'adjlist':
            g.read_adjlist(filename=self.input_file)
        elif graph_format == 'edgelist':
            g.read_edgelist(filename=self.input_file, weighted=config.weighted,
                            directed=config.directed)
        self.g = g
        self.adj = g.G.adj
        self.adj_mat = self.getAdj()
        self.encoder_list = str([self.g.node_size, self.n_emb])
        self.root_nodes = list(g.G.nodes)
        self.n_node = len(self.root_nodes)
        # print("reading graphs...")
        # self.n_node, self.graph = utils.read_edges(config.train_filename, config.test_filename)
        # self.root_nodes = [i for i in range(self.n_node)]
        #
        # print("reading initial embeddings...")
        # self.node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
        #                                                n_node=self.n_node,
        #                                                n_embed=config.n_emb)
        self.node_embed_init_g = utils.read_embeddings(filename=args.pretrain_filename,
                                                       n_node=self.n_node,
                                                       n_embed=self.n_emb)
        self.pretrain_file = args.pretrain_filename

        # construct or read BFS-trees
        self.trees = None
        if os.path.isfile(self.cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(self.cache_filename, 'rb')
            # pickle_file = config.cache_filename
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(self.cache_filename, 'wb')
            # pickle_file = config.cache_filename

            if config.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()

        # self.gen_all_score = self.generator.all_score

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        # self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.saver = tf.train.Saver()

        self.sess.run(self.init_op)
        self.build_discriminator()

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
        return adj

    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in list(self.adj[cur_node]):
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def build_discriminator(self):
        """initializing the discriminator"""
        with tf.variable_scope("discriminator"):
            encoder_layer_list = ast.literal_eval(self.encoder_list)
            self.discriminator = sdne_dis.SDNE(self.g, encoder_layer_list=encoder_layer_list,
                                               alpha=config.alpha, beta=config.beta, nu1=config.nu1, nu2=config.nu2,
                                               learning_rate=None)

    def train(self):
        # restore the model from the latest checkpoint if exists
        # checkpoint = tf.train.get_checkpoint_state(config.model_log)
        # if checkpoint and checkpoint.model_checkpoint_path and config.load_model:
        #     print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
        #     self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        # self.write_embeddings_to_file()
        # self.evaluation(self)

        # for gan&dis compare
        train_file = '/home/ming/gittest/EvalNE/evalne/tests/data/network.edgelist'

        G1 = pp.load_graph(train_file, directed=True)
        G, _ = pp.prep_graph(G1, relabel=True, del_self_loops=False)
        labels_ = pp.read_labels('/home/ming/gittest/EvalNE/evalne/tests/data/wiki/wiki_labels.txt', idx_mapping=_)
        traintest_split = EvalSplit()
        traintest_split.read_splits('/home/ming/gittest/EvalNE/examples/', 0, directed=True, verbose=True)
        # traintest_split.compute_splits(G, train_frac=0.6, owa=False)
        nre = NREvaluator(traintest_split, dim=128)
        #         nlp = LPEvaluator(traintest_split, dim=128)
        #         nce = NCEvaluator(G, labels_, 'citeseer', num_shuffles=2, traintest_fracs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        #                           trainvalid_frac=0.6, dim=128)

        print("start training...")
        X_gen = pp.read_node_embeddings(self.pretrain_file, traintest_split.TG.nodes, self.n_emb, ' ', 'MY_model')
        results_gen_nre = nre.evaluate_ne(data_split=traintest_split, X=X_gen, method='MY_model',
                                          edge_embed_method='hadamard')
        #         results_gen_nlp = nre.
        #         results_gen = nce.evaluate_ne(X_gen, method_name='DnnGAN')
        auc_gen = results_gen_nre.test_scores.auroc()
        #         auc_dis = results_dis.test_scores.auroc()
        fscore_gen = results_gen_nre.test_scores.f_score()
        #         fscore_dis = results_dis.test_scores.f_score()
        metric_name, vals = results_gen_nre.get_all()
        #         write_line = str([i.method+str(i.get_all())+'\n' for i in results_gen])+'\n'
        write_line1 = 'epoch {e}: {aucg}     {fscgen}  |||||||||||  +\n'.format(e='-1', aucg=auc_gen, fscgen=fscore_gen)
        #         write_line = str(metric_name)+'\n'+str(vals)
        with open('/home/ming/gittest/GraphGAN_数据半采样半SDNE/results/gen_dis_picture/gen_dis2.txt', 'a+')as fp:
            #             fp.writelines(write_line)
            fp.writelines(write_line1)

        results_g = list()
        results_d = list()
        for epoch in range(self.n_epoch):
            print("epoch %d" % epoch)

            # save the model
            # if epoch > 0 and epoch % config.save_steps == 0:
            #     self.saver.save(self.sess, config.model_log + "model.checkpoint")

            # D-steps
            # center_nodes = []
            # neighbor_nodes = []
            # labels = []
            for d_epoch in range(self.n_epoch_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                # if d_epoch % config.dis_interval == 0:
                look_back = self.g.look_back_list
                center_index = []
                neighbor_index = []
                center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                # labels = labels*5
                for i in center_nodes:  # 转化为SDNE索引
                    center_index.append(look_back.index(str(i)))
                for i in neighbor_nodes:
                    neighbor_index.append(look_back.index(str(i).strip()))
                list_sort = []
                for i in range(len(center_index)):
                    list_sort.append([center_index[i], neighbor_index[i], labels[i]])
                train_tuple = np.asarray(list_sort)
                # np.random.shuffle(list_sort)
                # center_nodes = np.squeeze(list_sort[:, :1])
                # neighbor_nodes = np.squeeze(list_sort[:, 1:2])
                # labels = np.squeeze(list_sort[:, 2:3])

                dis_embedding = self.discriminator.train(train_tuple)
                dis_vectors = {}
                for i, embedding in enumerate(dis_embedding):
                    dis_vectors[look_back[i]] = embedding
                self.dis_vectors = dict(sorted(dis_vectors.items(), key=lambda item: int(item[0])))
                self.discriminator.save_embeddings(self.dis_vectors, config.emb_filenames[1])
                with open(config.result_filename, mode="a+")as f:
                    f.writelines("epoch: %i   sub_epoch: %i" % (epoch, d_epoch))
                # self.evaluation_new(self.input_file, dis_vectors, self.n_emb)
            with open(config.result_filename, mode="a+")as f:
                f.writelines("epoch: %i **********************************" % epoch)
            # self.evaluation_new(self.input_file, self.dis_vectors, self.n_emb)
            # training
            # train_size = len(center_nodes)
            # start_list = list(range(0, train_size, config.batch_size_dis))
            # np.random.shuffle(start_list)
            # ret_dis = 0
            # for start in start_list:
            #     end = start + config.batch_size_dis
            #     ret, _ = self.sess.run((self.discriminator.loss, self.discriminator.d_updates),
            #                   feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
            #                              self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
            #                              self.discriminator.label: np.array(labels[start:end])})
            #     ret_dis += ret
            # print(d_epoch,"次：dis_loss",ret_dis)

            # G-steps
            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(self.n_epoch_gen):

                # SDNE算法中节点顺序与生成器不同
                look_back = self.g.look_back_list
                node_1_index = []
                node_2_index = []
                #                 if g_epoch % config.gen_interval == 0:
                node_1, node_2 = self.prepare_data_for_g()
                #                 embedding_matrix_gen = np.random.rand(len(node_1),  self.n_emb)
                #                 embedding_matrix_dis = np.random.rand(len(node_2),  self.n_emb)
                #                 for c in range(len(node_1)):
                #                     embedding_matrix_gen[int(c), :] = str_list_to_float(self.dis_vectors[str(node_1[c])])

                #                 for d in range(len(node_2)):
                #                     embedding_matrix_dis[int(d), :] = str_list_to_float(self.dis_vectors[str(node_2[d])])

                #                 with tf.Session() as sess1:
                #                     reward = reward.eval(sess1)
                #                     print(reward)

                for i in node_1:  # 转化为SDNE索引
                    node_1_index.append(look_back.index(str(i)))
                for i in node_2:
                    node_2_index.append(look_back.index(str(i)))
                list_sort = []
                for i in range(len(node_1)):
                    list_sort.append([node_1_index[i], node_2_index[i]])
                train_tuple = np.asarray(list_sort)

                reward = self.discriminator.train(train_tuple, for_g=True)

                # training
                #                 train_size = len(node_1)
                train_size = len(reward)
                start_list = list(range(0, train_size, config.batch_size_gen))
                np.random.shuffle(start_list)
                loss_gen = 0
                for start in start_list:
                    end = start + config.batch_size_gen
                    #                     for c in range(config.batch_size_gen):
                    #                         embedding_matrix_gen[int(c), :] = str_list_to_float(self.dis_vectors[str(node_1[start:end][c])])

                    #                     for d in range(config.batch_size_gen):
                    #                         embedding_matrix_dis[int(d), :] = str_list_to_float(self.dis_vectors[str(node_2[start:end][d])])

                    #                     dis_score = tf.reduce_sum(tf.multiply(embedding_matrix_gen, embedding_matrix_dis), axis=1)
                    #                     dis_score = tf.clip_by_value(dis_score, clip_value_min=-10, clip_value_max=10)
                    #                     reward = tf.math.log(1 + tf.math.exp(dis_score))
                    _, _loss_gen = self.sess.run((self.generator.g_updates, self.generator.loss),
                                                 feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                                            self.generator.node_neighbor_id: np.array(
                                                                node_2[start:end]),
                                                            self.generator.reward: np.array(reward[start:end])})
                    loss_gen += _loss_gen
                print("生成器:  epoch_gen-{0}  loss = {1}".format(g_epoch, loss_gen))
                gen_embedding = self.sess.run(self.generator.embedding_matrix)

                gen_vectors = {}
                for i, embedding in enumerate(gen_embedding):
                    gen_vectors[i] = embedding
                self.generator.save_embeddings(gen_vectors, self.output_file, dim=self.n_emb)
                self.discriminator
            # self.write_embeddings_to_file()
            # self.evaluation(self)

            # for gan&dis compare
            X_gen = pp.read_node_embeddings(self.output_file, traintest_split.TG.nodes, self.n_emb, ' ', 'MY_model')
            X_dis = pp.read_node_embeddings(config.emb_filenames[1], traintest_split.TG.nodes, self.n_emb, ' ',
                                            'MY_model')
            # results_gen = nlp.evaluate_ne(data_split=traintest_split, X=X_gen, method='MY_model', edge_embed_method='hadamard')
            #             results_gen = nce.evaluate_ne(X_gen, method_name='DnnGAN')
            results_gen_nre = nre.evaluate_ne(data_split=traintest_split, X=X_gen, method='MY_model',
                                              edge_embed_method='hadamard')

            # results_dis = nlp.evaluate_ne(data_split=traintest_split, X=X_dis, method='MY_model', edge_embed_method='hadamard')
            #             results_dis = nce.evaluate_ne(X_dis, method_name='DnnGAN')
            results_dis_nre = nre.evaluate_ne(data_split=traintest_split, X=X_dis, method='MY_model',
                                              edge_embed_method='hadamard')
            auc_gen = results_gen_nre.test_scores.auroc()
            auc_dis = results_dis_nre.test_scores.auroc()
            fscore_gen = results_gen_nre.test_scores.f_score()
            fscore_dis = results_dis_nre.test_scores.f_score()
            #             metric_name, vals = results_gen.get_all(precatk_vals=[0,500,1000,1500,2000,2500,3000])
            #             metric_name_dis, vals_dis = results_dis.get_all(precatk_vals=[0,500,1000,1500,2000,2500,3000])
            write_line1 = 'epoch {e}: {aucg}    {fscgen}   |||||||||||||||| {aucd}    {fscdis}+\n'.format(e=epoch,
                                                                                                          aucg=auc_gen,
                                                                                                          fscgen=fscore_gen,
                                                                                                          aucd=auc_dis,
                                                                                                          fscdis=fscore_dis)
            #             write_line = str(metric_name)+str(metric_name_dis)+'\n'+str(vals)+'||||'+str(vals_dis)+'\n'
            #             write_line = str([i.method+str(i.get_all())+'\n' for i in results_gen])+'||||'+str([i.method+str(i.get_all())+'\n' for i in results_dis])+'\n'
            with open('/home/ming/gittest/GraphGAN_数据半采样半SDNE/results/gen_dis_picture/gen_dis4.txt', 'a+')as fp:
                #                 fp.writelines(write_line)
                fp.writelines(write_line1)
        print("training completes")

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""

        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                pos = list(self.adj[str(i)])
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(neg))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""

        paths = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], 5, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        # reward = self.sess.run(self.discriminator.reward,
        #                        feed_dict={self.discriminator.node_id: np.array(node_1),
        #                                   self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree

        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        all_score = self.sess.run(self.generator.all_score)
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[
                    np.asarray(current_node, dtype=int), np.asarray(node_neighbor, dtype=int)]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(self.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

    @staticmethod
    def evaluation(self):
        results = []
        if config.app == "link_prediction":
            for i in range(2):
                lpe = lp.LinkPredictEval(
                    config.emb_filenames[i], config.test_filename, config.test_neg_filename, self.n_node, self.n_emb)
                result = lpe.eval_link_prediction()
                results.append(config.modes[i] + ":" + str(result) + "\n")

        with open(config.result_filename, mode="a+") as f:
            f.writelines(results)

    def evaluation_new(self, train_file, data, dim):
        mt.eval_NE(train_file, data, dim, config.result_filename)


if __name__ == "__main__":
    graph_gan = GraphGAN(parse_args())
    graph_gan.train()
