# by: Rohan Pandit 

import numpy as np
from sklearn.decomposition import RandomizedPCA
from run_lle import select_N_models 
from run_lle import calcAccumVar

import sys 

def main():
	protein = sys.argv[1]

	X = load_file(protein) 

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
	#models = np.arange(N)
	#np.savetxt("output/{protein}/pca/kept.txt".format(**locals()), models)
	#np.save("output/{protein}/pca/RMSD.npy".format(**locals()), RMSD[0,:])

	pca = RandomizedPCA(n_components=100, copy=False)	
	proj = pca.fit_transform(X)
	acc_var = calcAccumVar(pca.explained_variance_ratio_)
	
	np.savetxt("output/{protein}/pca/acc_var.txt".format(**locals()), acc_var)
	np.save("output/%s/pca/proj.npy" % protein, proj)
	np.save("output/{protein}/pca/proj2D.npy".format(**locals()), proj[:,:2])

def load_file(protein):
	f = open("../LSDMap/{protein}.xyz".format(**locals())).read().splitlines()
	
	n_models, n_features = map(int, f.pop(0).split())

	list_ = [line.split() for line in f]
	X = np.array(list_, ndmin=2, dtype="float32")
	X = X.reshape(n_models, n_features) 

	return X 



if __name__ == "__main__": main()
