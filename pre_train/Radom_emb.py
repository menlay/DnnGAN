import numpy as np

a = np.random.rand(5242,50)
np.savetxt("./CA.emb3",a)
print(a)