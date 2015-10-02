#!/usr/bin/env python

import sys
import numpy as np
from time import time

verbose = False

#fmin, fmax, nchan, maxDT = 400.00, 800.00, 4096, 8192  # Full band
fmin, fmax, nchan, maxDT = 593.75, 606.25, 128, 1024 # Test band
fs,df = np.linspace(fmin,fmax,nchan,endpoint=False,retstep=True)

def subDT(f,dF=df):
    "Get needed DT of subband to yield maxDT over entire band"
    loc = f**-2 - (f+dF)**-2
    glo = fmin**-2 - fmax**-2
    return np.ceil((maxDT-1)*loc/glo).astype(int)+1

A = None 
B = None
def buildAB(numCols,dtype=np.uint32):
    "A and B hold the states of successive iterations"
    global A,B
    numRowsA = (subDT(fs)).sum()
    numRowsB = (subDT(fs[::2],fs[2]-fs[0])).sum()
    A = np.zeros([numRowsA,numCols],dtype)
    B = np.zeros([numRowsB,numCols],dtype)


Q = None
def buildQ():
    "Q organizes memory locations of subtransforms for each iteration"
    global Q
    Q = []
    for i in range(int(np.log2(nchan))+1):
        needed = subDT(fs[::2**i],df*2**i)
        Q.append(np.cumsum(needed) - needed)


def prep(cols,dtype=np.uint32):
    "Prepares necessary matrices for FDMT"
    buildAB(cols,dtype)
    buildQ()


def fdmt(I,retDMT=False):
    "Computes DM Transform. If retDMT returns transform, else returns max sigma"
    if I.dtype.itemsize < 4: I = I.astype(np.uint32)
    if A is None or A.shape[1] != I.shape[1] or A.dtype != I.dtype or True:
        prep(I.shape[1],I.dtype)
     
    t1 = time()
    fdmt_initialize(I)
    
    t2 = time()
    for i in xrange(1,int(np.log2(nchan))+1):
        src, dest  = (A, B) if (i % 2 == 1) else (B, A)
        fdmt_iteration(src,dest,i)
    
    if verbose: 
        t3 = time()
        print "Initializing time:  %.2f s" % (t2-t1)
        print "Iterating time:  %.2f s" % (t3-t2)
        print "Total time: %.2f s" % (t3-t1)

    DMT = dest[:maxDT]
    if retDMT: return DMT
    noiseRMS  = np.array([DMT[i,i:].std()  for i in xrange(maxDT)])
    noiseMean = np.array([DMT[i,i:].mean() for i in xrange(maxDT)])
    sigmi = (rawDMT.T - noiseMean)/noiseRMS
    if verbose: print "Maximum sigma value: %.3f" % sigmi.max()
    return sigmi.max()


def fdmt_initialize(I):
    A[Q[0],:] = I
    chDTs     = subDT(fs)
    T         = I.shape[1]
    commonDTs = [T for _ in xrange(1,chDTs.min())] 
    DTsteps   = list(np.where(chDTs[:-1]-chDTs[1:] != 0)[0])
    DTplan    = commonDTs + DTsteps[::-1]
    for i,t in enumerate(DTplan,1):
        A[Q[0][:t]+i,i:] = A[Q[0][:t]+i-1,i:]+I[:t,:-i]
    for i,t in enumerate(DTplan,1):
        A[Q[0][:t]+i,i:] /= np.sqrt(float(i))
    

def fdmt_iteration(src,dest,i):
    T        = src.shape[1]
    dF       = df*2**i
    f_starts = fs[::2**i]
    f_ends   = f_starts + dF
    f_mids   = fs[2**(i-1)::2**i]
    for i_F in xrange(nchan/2**i):
        f0  = f_starts[i_F]
        f1  = f_mids[i_F]
        f2  = f_ends[i_F]
        C   = (f1**-2-f0**-2)/(f2**-2-f0**-2)
        cor = df/2
        C01 = ((f1-cor)**-2-f0**-2)/(f2**-2-f0**-2)
        C12 = ((f1+cor)**-2-f0**-2)/(f2**-2-f0**-2)
        for i_dT in xrange(subDT(f0,dF)):
            if i_dT*(C12-C01) > 1.1 and False:
                dT_mid01 = np.round(i_dT*C)
                dT_mid12 = np.floor(i_dT*C)
            else:
                dT_mid01 = np.ceil(i_dT*C)
                dT_mid12 = np.floor(i_dT*C)
            dT_rest = i_dT - dT_mid12
            assert(dT_rest >= 0)
            dest[Q[i][i_F]+i_dT,:] = src[Q[i-1][2*i_F]+dT_mid01,:]
            dest[Q[i][i_F]+i_dT,dT_mid12:] += \
                    src[Q[i-1][2*i_F+1]+dT_rest,:T-dT_mid12]


def recursive_fdmt(I,depth=0,curMax=0):
    "Performs FDMT, downsamples and repeats recursively, returning max sigma"
    curMax = max(curMax, fdmt(I))
    if depth <= 0: 
        return curMax
    else:
        I2  = I[:,::2]+I[:,1::2] if (I.shape[1]%2 == 0) else I[:,:-1:2]+I[:,1::2]
        return recursive_fdmt(I2,depth-1,curMax)


if __name__ == '__main__':
    if len(sys.argv) is not 3: 
        print "Usage: ./fdmt.py binary_image_file datatype"
    else:
        fn = sys.argv[1]
        dt = np.dtype(sys.argv[2])
        I  = np.fromfile(fn,dt)
        assert I.shape[0]%nchan == 0, 'Input shape inconsistent with decided nchan (%i)' % nchan 
        I  = I.reshape(nchan,I.shape[0]/nchan)
        print "Maximum sigma: %.2f" % recursive_fdmt(I)
