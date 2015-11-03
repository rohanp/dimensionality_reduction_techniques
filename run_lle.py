# by: Rohan Pandit

import sys
import numpy as np
from locally_linear import LocallyLinearEmbedding
from time import time

# method = 'standard', 'hessian', or 'ltsa' 
def main(protein, method):
	print("started script")
	RMSD = np.fromfile(open("../LSDMap/rmsd/%s.xyz_rmsd_all" % protein,'rb') , dtype="float32")
	N = int(np.sqrt(RMSD.shape[0]))
	RMSD = RMSD.reshape((N,N))
	"""	
	scores = np.loadtxt("../LSDMap/{protein}.scores.txt".format(**locals()))

	if scores.shape[0] != RMSD.shape[0] - 1:
		scores = scores[-RMSD.shape[0]:]
		print("selecting last N")

	models = select_N_models(RMSD[1:,1:], scores, 10000) 
	keep = np.r_[0, models + 1]"
	n_neigh = np.min(np.sum(RMSD < 6, axis=0)[models + 1])
	RMSD = RMSD[keep,:][:,keep]
	"""
	
	n_neigh = 20 #np.min(np.sum(RMSD < 6))

	if n_neigh < 20:
		n_neigh = 20
	if method == "modified":
		n_neigh = 20
	if method == "ltsa":
		n_neigh = 20

	print("num neighbors: ", n_neigh)
	models = np.arange(N) 
	np.savetxt("output/{protein}/LLE_{method}/kept.txt".format(**locals()), models)
	np.save("output/{protein}/LLE_{method}/RMSD.npy".format(**locals()), RMSD[0,:])

	t0 = time()

	lle = LocallyLinearEmbedding(n_neighbors=n_neigh, n_components=10,
				     eigen_solver="dense", method=method)

	proj = lle.fit_transform(RMSD)	

	evals = lle.evals_
	acc_var = calcAccumVar(evals[ 0 < evals ])
	
	np.save("output/{protein}/LLE_{method}/evecs.npy".format(**locals()), lle.evecs_)
	np.savetxt("output/{protein}/LLE_{method}/evals.txt".format(**locals()), evals)
	np.savetxt("output/{protein}/LLE_{method}/acc_var.txt".format(**locals()), acc_var)
	np.save("output/{protein}/LLE_{method}/proj.npy".format(**locals()), proj)
	np.save("output/{protein}/LLE_{method}/proj2D.npy".format(**locals()), proj[:,:2])
	
	print("finished! in ", round((time()-t0)/60,2) , "minutes")


def calcAccumVar(evals):
	accum = np.cumsum(evals)
	return accum / accum[-1]

def select_N_models(RMSD, scores, N):
	min_neigh = 20
	max_neigh = 60

	#remove outliers and tally num neighbors 	
	real_models = []
	n_neighbors_arr = []
	for i in range(RMSD.shape[0]):
		for cutoff in np.arange(1, 8, 0.25):
			n_neighbors = np.sum(RMSD[i] < cutoff)
			if n_neighbors < min_neigh:
				continue
			elif n_neighbors > max_neigh:
				break
			else:
				real_models.append(i)
				n_neighbors_arr.append(n_neighbors) 
				break

	print("num not outliers: ", len(real_models)) 
	n_neighbors_arr = np.asarray(n_neighbors_arr).astype(int)  
	real_models = np.asarray(real_models).astype(int)  
	RMSD = RMSD[real_models, :][:, real_models]  
	scores = scores[real_models] 	

	scale = np.arange(np.min(scores), np.max(scores), 5) 
	bins = np.digitize(scores, scale) #what bin each conf belongs to

	# find the K models with highest local density in each bin
	K = int(N / scale.shape[0])

	top_models = []

	while(len(top_models) < N):
		for bin in range(scale.shape[0]):
			if len(np.where(bins == bin)[0]) != 0:

				models_in_bin = np.where(bins == bin)
				neighbors = n_neighbors_arr[models_in_bin]  	
				# choose K best models, evaled by num neighbors 
				top_models  += list(real_models[ np.argsort(neighbors)[::-1][:K] ])

				if len(top_models) > N:
					break 

	return np.array(top_models)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])

"""
for protein in ('1AOY', '1BQ9', '1C8CA', '1DTDB', '1HHP', '1PGB', '1WAPA'):
	try:
		main(protein)
	except Exception:
		print("something went wrong")
"""
