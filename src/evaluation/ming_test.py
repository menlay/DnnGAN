import sys
sys.path.append("/home/ming/gittest/GraphGAN_数据半采样半SDNE/src/evaluation")
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
from evalne.utils import preprocess as pp
import scipy.io as scio
import numpy as np


def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """
    a = filename.strip().split('.')
    if a[-1]=='emb':
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # skip the first line
            embedding_matrix = np.random.rand(n_node, n_embed)
            for line in lines:
                emd = line.split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    else:
        embedding_matrix = np.genfromtxt(filename,delimiter=' ')

    return embedding_matrix


def eval_NE(train_file,data,dim,result_file):
    G = pp.load_graph(train_file)
    G, _ = pp.prep_graph(G)
    traintest_split = EvalSplit()
    # train_E, train_E_neg, test_E, test_E_neg = traintest_split.compute_splits(G, split_alg='random', fe_ratio=1)
    # traintest_split.set_splits(train_E, train_E_false=train_E_neg, test_E=test_E, test_E_false=test_E_neg, split_id=0)
    traintest_split.compute_splits(G)

    nee = LPEvaluator(traintest_split,dim)

    scoresheet = Scoresheet()

    # Set the baselines
    methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient']
    try:
        nodes = np.arange(0, len(G.nodes), 1)
        # embedding = scio.loadmat('examples/ming/embedding.mat')
        # data = pp.read_node_embeddings('examples/ming/CA-GrQc_dis_100e100w1.0.txt', nodes, 100, delimiter=',')
        result = nee.evaluate_ne(traintest_split, data, "GG_SDNE", "average")
        result.params['eval_time'] = np.float(2)
        scoresheet.log_results(result)
        # result = nee.compute_results(traintest_split,'GG_SDNE',train_pred,test_pred)

    except ImportError:
        print("The OpenNE library is not installed. Reporting results only for the baselines...")
        pass
    scoresheet.print_tabular(metric='f_score')
    scoresheet.write_all(result_file)
    #return scoresheet.get_pandas_df()
# # embedding = read_embeddings('examples/ming/CA-GrQc_dis_.emb',5242,50)
# # scio.savemat('examples/ming/CA-GrQc_dis_.mat', {'A': embedding})
# # Load and preprocess the network
# G = pp.load_graph('examples/ming/CA-GrQc_train.txt')
# G, _ = pp.prep_graph(G)
# # train_E = np.loadtxt('examples/ming/CA-GrQc_train.txt', delimiter=',', dtype=int)
# # test_E = np.loadtxt('examples/ming/CA-GrQc_test.txt', delimiter=',', dtype=int)
# # # test_E = np.append(test_E,train_E[1449:,:],axis=0)
# # train_E_neg, test_E_neg, G = pp.relabel_nodes(train_E, test_E, directed=0)
# # # Create an evaluator and generate train/test edge split
# # traintest_split = EvalSplit()
# # # traintest_neg_split = np.loadtxt('examples/ming/CA-GrQc_test_neg1.txt', delimiter=',', dtype=int)
# # # test_edg = np.loadtxt('examples/ming/CA-GrQc_test.txt', delimiter=',', dtype=int)
# # # test_neg_edg = np.loadtxt('examples/ming/CA-GrQc_test_neg1.txt', delimiter=',', dtype=int)
# # # train_E = np.loadtxt('examples/ming/CA-GrQc_train.txt', delimiter=',', dtype=int)
#
# traintest_split = EvalSplit()
# train_E, train_E_neg,test_E, test_E_neg =traintest_split.compute_splits(G,split_alg='random',fe_ratio=1)
# traintest_split.set_splits(train_E,train_E_false=train_E_neg,test_E=test_E, test_E_false=test_E_neg,split_id=0)
# nee = LPEvaluator(traintest_split,dim=100)
#
# # Create a Scoresheet to store the results
# scoresheet = Scoresheet()
#
# # Set the baselines
# methods = ['random_prediction', 'common_neighbours', 'jaccard_coefficient']
#
# # Evaluate baselines
# # for method in methods:
# #     result = nee.evaluate_baseline(method=method)
# #     scoresheet.log_results(result)
#
# try:
#     # Check if OpenNE is installed
#     # import openne
#     #
#     # # Set embedding methods from OpenNE
#     # methods = ['node2vec', 'deepwalk', 'GraRep']
#     # commands = [
#     #     'python -m openne --method node2vec --graph-format edgelist --p 1 --q 1',
#     #     'python -m openne --method deepWalk --graph-format edgelist --number-walks 40',
#     #     'python -m openne --method grarep --graph-format edgelist --epochs 10']
#     # edge_emb = ['average', 'hadamard']
#     #
#     # # Evaluate embedding methods
#     # for i in range(len(methods)):
#     #     command = commands[i] + " --input {} --output {} --representation-size {}"
#     #     results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
#     #                                edge_embedding_methods=edge_emb, input_delim=' ', output_delim=' ')
#     #     scoresheet.log_results(results)
#     # data = scio.loadmat('examples/ming/CA-GrQc_dis_.mat')
#     # train_pred, test_pred = nee.compute_pred(traintest_split,data['A'])
#     nodes = np.arange(0,5242,1)
#     # embedding = scio.loadmat('examples/ming/embedding.mat')
#     data = pp.read_node_embeddings('examples/ming/CA-GrQc_dis_100e100w1.0.txt', nodes, 100, delimiter=',')
#     result = nee.evaluate_ne(traintest_split, data, "GG_SDNE", "average")
#     result.params['eval_time']= '2'
#     scoresheet.log_results(result)
#     # result = nee.compute_results(traintest_split,'GG_SDNE',train_pred,test_pred)
#
# except ImportError:
#     print("The OpenNE library is not installed. Reporting results only for the baselines...")
#     pass
#
# # Get output
# scoresheet.print_tabular()