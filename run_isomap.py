# by: Rohan Pandit

import sys
import numpy as np
from calc_isomap import Isomap
from time import time 
#from run_lle import select_N_models

def main(protein):
	print("started script")
	RMSD = np.fromfile(open("../LSDMap/rmsd/%s.xyz_rmsd_all" % protein,'rb') , dtype="float32")
	N = int(np.sqrt(RMSD.shape[0]))
	RMSD = RMSD.reshape((N,N))
	"""
	scores = np.loadtxt("../LSDMap/{protein}.scores.txt".format(**locals()))

	if scores.shape[0] != RMSD.shape[0]:
		scores = scores[-RMSD.shape[0]:]
		print("selecting last N")

	models = select_N_models(RMSD[1:,1:], scores, 10000)
	keep = np.r_[0, models + 1]
	n_neigh = np.min(np.sum(RMSD < 6, axis=0)[models + 1])
	RMSD = RMSD[keep,:][:,keep]
	"""
	
	n_neigh = np.min(np.sum(RMSD < 6, axis=0)) 

	if n_neigh < 20:
		n_neigh = 20
	print("num neighbors: ", n_neigh)

	models = np.arange(N)
	np.savetxt("output/{protein}/Isomap/kept.txt".format(**locals()), models)
	np.save("output/{protein}/Isomap/RMSD.npy".format(**locals()), RMSD[0,:])
	t0 = time()
	
	iso = Isomap(n_neighbors=n_neigh, n_components=10,
		        eigen_solver="dense")

	proj = iso.fit_transform(RMSD)	

	evals = np.sort(iso.kernel_pca_.evals_)[::-1]
	evecs = iso.kernel_pca_.evecs_
	print(evals.shape)
	print(evecs.shape)  
	acc_var = calcAccumVar(evals[ 0 < evals]) 
	
	np.save("output/{protein}/Isomap/evecs.npy".format(**locals()), evecs) 
	np.savetxt("output/{protein}/Isomap/acc_var.txt".format(**locals()), acc_var)
	np.savetxt("output/{protein}/Isomap/evals.npy".format(**locals()), evals)
	np.save("output/{protein}/Isomap/proj.npy".format(**locals()), proj)
	np.save("output/{protein}/Isomap/proj2D.npy".format(**locals()), proj[:,:2])

	print("finished! in ", (time()-t0)/60, "minutes")


def calcAccumVar(evals):
	accum = np.cumsum(evals)
	return accum / accum[-1]

if __name__ == "__main__":
	main(sys.argv[1])

