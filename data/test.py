import scipy.io as scio

data_file = './Homo_sapiens.mat'
data = scio.loadmat(data_file)
print(data)