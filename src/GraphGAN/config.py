modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-4  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 5  # number of samples for the generator
lr_gen = 1e-5  # learning rate for the generator
lr_dis = 0.0001  # learning rate for the discriminator
n_epochs = 1 # number of outer loops
n_epochs_gen = 30  # number of inner loops for the generator
n_epochs_dis = 30  # number of inner loops for the discriminator
gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1    # updating ratio when choose the trees

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
n_emb = 128
multi_processing = True  # whether using multi-processing to construct BFS-trees
window_size = 2

# application and dataset settings
app = "link_prediction"
dataset = "karate"

# path settings
# train_filename = "../../data/" + app + "/" + dataset + "_train.txt"
train_filename = "../../data/karate/karate.edgelist"
test_filename = "../../data/" + app + "/" + dataset + "_test.txt"
test_neg_filename = "../../data/" + app + "/" + dataset + "_test_neg.txt"
pretrain_emb_filename_d = "../../pre_train/" + app + "/" + dataset + "_pre_train.emb"
pretrain_emb_filename_g = "/home/ming/gittest/GraphGAN_数据半采样半SDNE/pre_train/link_prediction/karate_pre_train.emb"
# pretrain_emb_filename_d = "/home/ming/gittest/GraphGAN/pre_train/CA.emb2"
# pretrain_emb_filename_g = "/home/ming/gittest/GraphGAN/pre_train/CA.emb3"
emb_filenames = ["../../results/" + app + "/" + dataset + "_gen_.emb",
                 "../../results/" + app + "/" + dataset + "_dis_.emb"]
result_filename = "../../results/" + app + "/" + dataset + ".txt_G_D"
cache_filename = "../../cache/" + dataset + ".pkl"
model_log = "../../log/"


weighted = False
directed = False
alpha = 1e-06
beta = 5.0
nu1 = 1e-05
nu2 = 0.0001

