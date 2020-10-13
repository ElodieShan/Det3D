import numpy as np
a = np.array([[1,2,3,4],
              [2,2,2,2],
              [3,3,3,3],
              [4,4,4,4]])
a_mask = np.any(a[:,-1] in [4],axis=0)