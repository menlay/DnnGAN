import numpy as np
import scipy.io as scio

def str_list_to_float(str_list):
    return [float(item) for item in str_list]
def emb_to_txt():
    with open ('/home/ming/gittest/EvalNE/examples/ming/CA-GrQc_dis_100e100w1.0.emb', "r")as tf:
        first_line = tf.readline()
        nodes, dim = first_line.split()
        lines = tf.readlines()
        embedding = np.random.rand(int(nodes),int(dim)+1)
        for line in lines:
            emb = line.split()
            embedding[int(emb[0]), :] = str_list_to_float(emb)

        np.savetxt("/home/ming/gittest/EvalNE/examples/ming/CA-GrQc_dis_100e100w1.0.txt",embedding)

emb_to_txt()
# emb = scio.loadmat("/home/ming/gittest/EvalNE/examples/ming/embedding_sdne.mat")
# embedding = emb['embedding']
# nodes = np.arange(5242)
# embedding = np.insert(embedding,0,nodes,axis=1)
# np.savetxt("/home/ming/gittest/EvalNE/examples/ming/CA-GrQc_sdne_.emb",embedding)