import pickle
import numpy as np

x = np.array([0.99, 1.27, 0.21, 0.70710678, 0.70710678, 0., 0.])
p = pickle.load(open('demonstrations/d9-block3.pickle','rb'))

for d in range(len(p)):
	for i in p[d]['observations'].shape[0]:
		y =  p[d]['observations'][i].copy()
		import pdb
		pdb.set_trace()
		p[d]['observations'][i] = np.append(y[:30], x)


pickle.dump(p, open('new-dec-block5.pickle','wb'))