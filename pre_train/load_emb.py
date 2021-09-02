# from .mio4 import MatFile4Reader, MatFile4Writer
# from .mio5 import MatFile5Reader, MatFile5Writer
import scipy.io as sio
import numpy as np
import numpy as np
path = "/home/ming/gittest/SDNE/result/ca-Grqc-Sun-May-31-17:28:20-2020/embedding.mat"
Nodes = []
with open("/home/ming/gittest/SDNE/GraphData/ca-Grqc.txt", "r+") as fp:
    first_line = fp.readline()
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split()
        Nodes.append(line[0])

load_data = sio.loadmat(path)
# for line in load_data:
#     print(line)
# print(type(load_data))
# print(load_data.keys())
# print(load_data.values())
# print(load_data["embedding"])
# load_data["embedding"]

#np.savetxt("./CA.emb1",load_data["embedding"])
# data1 = np.loadtxt("./CA.emb1",load_data[])
# data2 = np.insert(data1,0,values=Nodes,axis=1)
np.savetxt("./CA.emb2",load_data["embedding"])
# with open(path,"rb") as fb:
#     for line in fb:
#         print(line)[]