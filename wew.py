# import os.path
# from jupyter_core.paths import jupyter_path
#
# def find_extension(ext_path):
#     for base in jupyter_path():
#         curr_path = os.path.join(base, 'nbextensions', ext_path)
#         if os.path.exists(curr_path):
#             print('found extension at {!r}'.format(curr_path))
#
# find_extension('config/main.js')

import numpy as np
d = []
while True:
    d.append(np.zeros(shape=(1000,1000)))