#cython: wraparound=False, boundscheck=False, cdivision=True
#cython: profile=False, nonecheck=False, overflowcheck=False
#cython: cdivision_warnings=True, unraisable_tracebacks=False
 
""" A Python/Cython implementation of the Locally-Scaled Diffusion Map 
	Dimensionality Reduction Technique.  
"""
__author__ = "Rohan Pandit" 
 
import numpy as np
cimport numpy as np
from time import time
from libc.math cimport sqrt, exp
import random

def calcMarkov(RMSDs, epsilons):
	print(RMSDs.dtype)
	print(epsilons.dtype)
	return _calcMarkovMatrix(RMSDs, epsilons, RMSDs.shape[0])

cdef double[:,:] _calcMarkovMatrix(float[:,:] RMSD, double[:] epsilons, int N):	
	cdef: 
		int i, j
		double[:] D = np.zeros(N)
		double[:] Dtilda = np.zeros(N)
		double[:,:] K = np.zeros((N,N))
		double[:,:] Ktilda = np.zeros((N,N))
		double[:,:] P = np.zeros((N,N))

	with nogil:
		for i in range(N):
			for j in range(N):
				K[i, j] = exp( (-RMSD[i, j] * RMSD[i, j]) / (2*epsilons[i] * epsilons[j]) )
				D[i] += K[i, j]

		for i in range(N):
			for j in range(N):
				Ktilda[i, j] = K[i, j] / sqrt(D[i]*D[j])
				Dtilda[i] += Ktilda[i, j]

		for i in range(N):
			for j in range(N):
				P[i, j] = Ktilda[i, j] / Dtilda[i]

	return P

