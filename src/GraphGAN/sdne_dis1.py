
import numpy as np
import tensorflow as tf
import time
import copy
import random
import config
tf.disable_eager_execution()

def fc_op_1(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer()([n_in, n_out]), dtype=tf.float32, name=scope + "d1_w")
        # kernel = tf.Variable(tf.random_normal([n_in, n_out]), name=scope + "w")

        # kernel = tf.Variable(tf.random_normal([n_in, n_out]))
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'd1_b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'd1_act')
        layer_collector.append([kernel, biases])
        return activation

def fc_op(input_op, input_center_op, imput_neighbor_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer()([n_in, n_out]), dtype=tf.float32, name=scope + "d_w")
        # kernel = tf.Variable(tf.random_normal([n_in, n_out]), name=scope + "w")
        # kernel = tf.Variable(tf.random_normal([n_in, n_out]))
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'd_b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        center_fc = tf.add(tf.matmul(input_center_op, kernel), biases)
        neighbor_fc = tf.add(tf.matmul(imput_neighbor_op, kernel), biases)

        activation = act_func(fc, name=scope + 'd_act')
        activation_1 = act_func(center_fc, name=scope + 'd_act_center')
        activation_2 = act_func(neighbor_fc, name=scope + 'd_act_neighbor')
        layer_collector.append([kernel, biases])
        return activation, activation_1, activation_2



class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=1e-6, beta=5.0, nu1=1e-5, nu2=1e-4,
                 learning_rate=None):
        """
        encoder_layer_list: a list of numbers of the neuron at each ecdoer layer, the last number is the
        dimension of the output node representation
        Eg:
        if node size is 2000, encoder_layer_list=[1000, 128], then the whole neural network would be
        2000(input)->1000->128->1000->2000, SDNE extract the middle layer as the node representation
        """
        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.dim = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        # self.epoch = epoch
        #self.max_iter = (epoch * self.node_size) // batch_size
        self.max_iter = config.n_epochs*config.n_epochs_gen*self.node_size*10

        self.lr = learning_rate
        self.iter = 0  # 当前循环次数  用来计算lr
        self.global_step = tf.Variable(tf.constant(0))
        if self.lr is None:
            self.lr = tf.train.inverse_time_decay(0.001, self.global_step, decay_steps=1, decay_rate=0.5)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.vectors = {}

        self.adj_mat = self.getAdj()
        # self.embeddings = self.train()
        #
        # look_back = self.g.look_back_list
        #
        # for i, embedding in enumerate(self.embeddings):
        #     self.vectors[look_back[i]] = embedding

        #for train
         # adj_mat = self.adj_mat

        self.AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        self.center_AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='center_adj_batch')
        self.neighbor_AdjBatch = tf.placeholder(tf.float32, [None, self.node_size], name='neighbor_adj_batch')
        self.Adj = tf.placeholder(tf.float32, [None, None], name='adj_mat')
        self.B = tf.placeholder(tf.float32, [None, self.node_size], name='b_mat')
        self.label = tf.placeholder(tf.float32, shape=[None])


        center_fc= self.center_AdjBatch
        neighbor_fc = self.neighbor_AdjBatch
        fc = self.AdjBatch

        scope_name = 'encoder'
        layer_collector = []

        # with tf.name_scope(scope_name):
        #     for i in range(1, self.encoder_layer_num):
        #         fc = fc_op(fc,
        #                    name=scope_name + str(i),
        #                    n_out=self.encoder_layer_list[i],
        #                    layer_collector=layer_collector)

        with tf.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                fc, center_fc, neighbor_fc = fc_op(fc, center_fc, neighbor_fc,
                           name=scope_name + str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        self._embeddings = fc
        self.center_embedding = center_fc
        self.neighbor_embedding = neighbor_fc
        self.score = tf.reduce_sum(tf.multiply(self.center_embedding, self.neighbor_embedding), axis=1)

        scope_name = 'decoder'
        with tf.name_scope(scope_name):
            for i in range(self.encoder_layer_num - 2, 0, -1):
                fc = fc_op_1(fc,
                           name=scope_name + str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op_1(fc,
                       name=scope_name + str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector, )

        self._embeddings_norm = tf.reduce_sum(tf.square(self._embeddings), 1, keepdims=True)

        L_1st = tf.reduce_sum(
            self.Adj * (
                    self._embeddings_norm - 2 * tf.matmul(
                self._embeddings, tf.transpose(self._embeddings)
            ) + tf.transpose(self._embeddings_norm)
            )
        )
        self.L_1st = L_1st
        L_2nd = tf.reduce_sum(tf.square((self.AdjBatch - fc) * self.B))
        self.L_2nd = L_2nd
        L_dis = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) #+ config.lambda_dis * (tf.nn.l2_loss(self.neighbor_embedding) + tf.nn.l2_loss(self.center_embedding))
        self.L_dis = L_dis
#         self.L = L_2nd + 0.0001*(L_1st) + 0.001*L_dis
        self.L = L_2nd + L_dis

        for param in layer_collector:
            self.L += self.nu1 * tf.reduce_sum(tf.abs(param[0])) + self.nu2 * tf.reduce_sum(tf.square(param[0]))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.train_op = self.optimizer.minimize(self.L)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.math.log(1 + tf.math.exp(self.score))


        init = tf.global_variables_initializer()
        self.sess.run(init)
        # self.init_embedding = self.sess.run(self._embeddings, feed_dict={self.AdjBatch: self.adj_mat})

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
        return adj

    def train(self, train_tuple,for_g=False):
        if for_g:
            node_1 = np.squeeze(train_tuple[:, :1])
            node_2 = np.squeeze(train_tuple[:, 1:2])
            cpt_size = len(node_1)
            cpt_list = list(range(0,cpt_size,config.batch_size_dis))   #计算reward分批
            # cpt_max_iter = cpt_size // config.batch_size_dis  # batch_size=128
            # print("total iter: %i" % cpt_max_iter)
            reward = []
            for cpt in cpt_list:
                end = cpt+config.batch_size_dis

                node_1_t = np.asarray(node_1[cpt:end], dtype=np.int)
                node_2_t = np.asarray(node_2[cpt:end], dtype=np.int)
                adj_center_train = self.adj_mat[node_1_t, :]  # 选中节点的邻接[None,N]
                adj_neighbor_train = self.adj_mat[node_2_t, :]  # 选中节点的邻接[None,N]
                reward_bch = self.sess.run(self.reward, feed_dict={
                                                        self.center_AdjBatch: adj_center_train,
                                                        self.neighbor_AdjBatch: adj_neighbor_train,
                                                        })
                reward = np.append(reward,reward_bch,axis=0)
            return reward
        else:
            self.iter += 1
            center_index = np.squeeze(train_tuple[:, :1])
            neighbor_index = np.squeeze(train_tuple[:, 1:2])
            labels = np.squeeze(train_tuple[:, 2:3])
            max_iter = 2 * len(center_index) // config.batch_size_dis
            print("total iter: %i" % max_iter)
            for step in range(max_iter):

                index_1 = np.random.randint(self.node_size, size=int(config.batch_size_dis/2)) #随机采样
                index_2 = np.random.randint(len(center_index), size=config.batch_size_dis) #popred data for d 随机采样的索引
                center_index_t = center_index[index_2]
                neighbor_index_t = neighbor_index[index_2]
                index_mix = np.append(index_1, center_index_t[:int(config.batch_size_dis/2)])

                adj_batch_train = self.adj_mat[index_mix, :]
                adj_mat_train = adj_batch_train[:, index_mix]
                b_mat_train = np.ones_like(adj_batch_train)
                b_mat_train[adj_batch_train != 0] = self.beta

                # intersec = np.intersect1d(center_index,index_mix)  #两数组交集
                # select_index = []
                # if len(intersec)>0:
                #     for i in intersec:
                #         select_index = np.append(select_index,np.argwhere(center_index==i))
                #     select_index = select_index.astype(np.int).tolist()
                #     if len(select_index) > config.batch_size_dis:
                #         select_index = random.sample(select_index, int(config.batch_size_dis))
                    # else:
                    #     select_index = random.sample(select_index, len(select_index))
                    # select_index.append(np.argwhere(center_index==i))
                # else:
                #     select_index = np.random.randint(len(center_index), size=1)

                # center_index_t = center_index[select_index]
                # neighbor_index_t = neighbor_index[select_index]
                label_t = np.asarray(labels[index_2], dtype=np.float)
                adj_center_train = np.asarray(self.adj_mat[center_index_t, :], dtype=np.float)  #选中节点的邻接[None,N]
                adj_neighbor_train = np.asarray(self.adj_mat[neighbor_index_t, :], dtype=np.float)
                feed_dict = {self.AdjBatch: adj_batch_train,
                             self.center_AdjBatch: adj_center_train,
                             self.neighbor_AdjBatch: adj_neighbor_train,
                             self.Adj: adj_mat_train,
                             self.B: b_mat_train,
                             self.label: label_t,
                             self.global_step: self.iter}
                self.sess.run(self.train_op,feed_dict=feed_dict)
                if step % 50 == 0:
                    l, l1, l2, ldis, lr = self.sess.run((self.L, self.L_1st, self.L_2nd,self.L_dis, self.lr), feed_dict=feed_dict)
                    print("step %i: total loss: %s, l1 loss: %s, l2 loss: %s, ldis loss: %s, lr: %s" % (step, l, l1, l2, ldis, lr))

            return self.sess.run(self._embeddings, feed_dict={self.AdjBatch: self.adj_mat})


            # labels = np.squeeze(train_tuple[:, 2:3])
            # train_size = len(center_index)
            # start_list = list(range(0,train_size,config.batch_size_dis))
            # self.max_iter = train_size // config.batch_size_dis #batch_size=128
            # print("total iter: %i" % self.max_iter)
            # step = 0
            # for start in start_list:
            #     step+=1
            #     end = start + config.batch_size_dis
            #     # index = np.random.randint(self.node_size, size=self.bs)
            #     # index = np.append(center_nodes, neighbor_nodes)    #去重复，所有节点
            #     center_index_t = np.asarray(center_index[start:end],dtype=np.int)
            #     neighbor_index_t = np.asarray(neighbor_index[start:end],dtype=np.int)
            #     label = labels[start:end]
            #     index = np.append(center_index_t,neighbor_index_t)
            #     adj_batch_train = self.adj_mat[index, :]
            #     adj_center_train = self.adj_mat[center_index_t, :]    #选中节点的邻接[None,N]
            #     adj_neighbor_train = self.adj_mat[neighbor_index_t, :]    #选中节点的邻接[None,N]
            #     adj_mat_train = adj_batch_train[:, index]   #选中节点的邻接矩阵
            #     b_mat_train = np.ones_like(adj_batch_train)
            #     b_mat_train[adj_batch_train != 0] = self.beta   #增加非零节点对代价
            #
            #     self.sess.run(self.train_op, feed_dict={self.AdjBatch: adj_batch_train,
            #                                              self.center_AdjBatch: adj_center_train,
            #                                              self.neighbor_AdjBatch: adj_neighbor_train,
            #                                              self.Adj: adj_mat_train,
            #                                              self.B: b_mat_train,
            #                                              self.label: label})
            #     if step % 50 == 0:
            #         l, l1, l2, ldis = self.sess.run((self.L, self.L_1st, self.L_2nd,self.L_dis),
            #                                   feed_dict={self.AdjBatch: adj_batch_train,
            #                                              self.center_AdjBatch: adj_center_train,
            #                                              self.neighbor_AdjBatch: adj_neighbor_train,
            #                                              self.Adj: adj_mat_train,
            #                                              self.B: b_mat_train,
            #                                              self.label: label})
            #         print("step %i: total loss: %s, l1 loss: %s, l2 loss: %s, ldis loss: %s" % (step, l, l1, l2, ldis))
            #
            # return self.sess.run(self._embeddings, feed_dict={self.AdjBatch: self.adj_mat})

    def save_embeddings(self, vectors, filename):
        fout = open(filename, 'w')
        node_num = len(vectors)
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
