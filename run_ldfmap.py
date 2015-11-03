#by: Rohan Pandit

import sys
import numpy as np
from calcMarkov import calcMarkov
from run_lle import select_N_models, calcAccumVar
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import os

def main(protein):
	print("started script")
	RMSD = np.fromfile(open("../LSDMap/rmsd/%s.xyz_rmsd_all" % protein,'rb') , dtype="float32")
	epsilons = np.array([0.3*np.max(RMSD)] * RMSD.shape[0], dtype="float64") #np.loadtxt("../LSDMap/%s.epsilons.txt" % protein)
	N = int(np.sqrt(RMSD.shape[0]))
	RMSD = RMSD.reshape((N,N))
	"""
	scores = np.loadtxt("../LSDMap/{protein}.scores.txt".format(**locals()))
	
	if scores.shape[0] != RMSD.shape[0] - 1:
		scores = scores[-RMSD.shape[0]:]
		print("selecting last N")


	models = select_N_models(RMSD[1:,1:], scores, 7500)
	keep = np.r_[0, models + 1]
	n_neigh = np.min(np.sum(RMSD < 6, axis=0)[models + 1])
	RMSD = RMSD[keep,:][:,keep]
	
	np.savetxt("output/{protein}/LDFMap/kept.txt".format(**locals()), models)
	np.save("output/%s/LDFMap/RMSD.npy" % protein, RMSD[0,:])
	"""
	print("starting markov")
	print(RMSD.shape)
	P = calcMarkov(RMSD, epsilons)
	np.save("output/%s/LDFMap/markov.npy" % protein , P)

	del RMSD 
	P = np.asarray(P)

	#evals, evecs = eigsh(P, 100, which='LA') 
	#evals = evals[::-1] 
	#evecs = evecs[:, ::-1]

	evals, evecs = eigh(P) 
	order = np.argsort(evals)[::-1]
	evals = evals[order]
	evecs = evecs[:, order] 
	
	acc_var = calcAccumVar(evals) 

	np.savetxt("output/{protein}/LDFMap/acc_var.txt".format(**locals()), acc_var)
	np.save("output/%s/LDFMap/evecs.npy" % protein, evecs) 
	np.savetxt("output/%s/LDFMap/evals.txt" % protein, evals)

	print("starting projection")
	projections = np.dot(P, evecs) 

	np.save("output/%s/LDFMap/proj.npy" % protein, projections)
	np.save("output/%s/LDFMap/proj2D.npy" % protein, projections[:,:2])

if __name__ == "__main__":
	main(sys.argv[1])

