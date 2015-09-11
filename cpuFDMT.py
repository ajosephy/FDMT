#!/usr/bin/env python

import sys
import numpy as np
from time import time

verbose = False
doWeighting = True

#fmin, fmax, nchan, maxDT = 400.00, 800.00, 4096, 8192  # Full band
fmin, fmax, nchan, maxDT = 593.75, 606.25, 128, 256  # Test band
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


def fdmt(I,returnMaxSigma=True):
    "Computes DM Transform"
    if I.dtype.itemsize < 4: I = I.astype(np.uint32)
    
    if A is None or A.shape[1] != I.shape[1] or A.dtype != I.dtype:
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
    #return DMT.max()
    if returnMaxSigma:
        maxSigma = 0
        maxRow = 0
        r1 = DMT[0].std()
        r0 = DMT[0].mean()
        for i in xrange(maxDT):
            r = DMT[i,i:]
            R = (r.max()-r.mean())/r.std()
            if R > maxSigma:
                maxSigma = R
                maxRow = i
            #maxSigma = max(maxSigma,r.std()) 
        if verbose: print "Maximum sigma value: %.3f" % maxSigma
        return maxSigma #DMT[maxRow,maxRow:].std()
    else:
        return dest[:maxDT]


def fdmt_initialize(I):
    chDTs = subDT(fs)
    DTsteps = np.where(chDTs[:-1]-chDTs[1:] != 0)[0]
    A[Q[0],:] = I
    A[Q[0]+1,1:] = A[Q[0],1:] + I[:,:-1]
    for i,s in enumerate(DTsteps[::-1]):
        A[Q[0][:s]+i+2,i+2:] = A[Q[0][:s]+i+1,i+2:] + I[:s,:-i-2]
    
    if doWeighting:
        A[Q[0]+1,1:] /= np.sqrt(2.)
        for i,s in enumerate(DTsteps[::-1]):
            A[Q[0][:s]+i+2,i+2:] /= np.sqrt(float(i+2))
    

def fdmt_iteration(src,dest,i):
    T        = A.shape[1]
    dF       = df*2**i
    f_starts = fs[::2**i]
    f_ends   = f_starts + dF
    f_mids   = fs[2**(i-1)::2**i]
    for i_F in xrange(nchan/2**i):
        f0  = f_starts[i_F]
        f1  = f_mids[i_F]
        f2  = f_ends[i_F]
        C = (f1**-2-f0**-2)/(f2**-2-f0**-2)
        cor = df/16
        C01 = ((f1-cor)**-2-f0**-2)/(f2**-2-f0**-2)
        C12 = ((f1+cor)**-2-f0**-2)/(f2**-2-f0**-2)   
        for i_dT in xrange(subDT(f0,dF)):
            dT_mid01 = round(i_dT * C01) if i_dT > subDT(f0,dF)/2 else round(i_dT * C)
            dT_mid12 = round(i_dT * C12) if i_dT > subDT(f1,dF)/2 else round(i_dT * C)
            dT_rest = i_dT - dT_mid12
            dest[Q[i][i_F]+i_dT,:] = src[Q[i-1][2*i_F]+dT_mid01,:]
            dest[Q[i][i_F]+i_dT,dT_mid12:] += \
                    src[Q[i-1][2*i_F+1]+dT_rest,:T-dT_mid12]


def recursive_fdmt(I,curMax=0,depth=2):
    "Performs FDMT, downsamples and repeats recursively, returning max sigma"
    curMax = max(curMax, fdmt(I,returnMaxSigma=True))
    if depth <= 0: return curMax
    I2  = I[:,::2]+I[:,1::2] if (I.shape[1]%2 == 0) else I[:,:-1:2]+I[:,1::2]
    if doWeighting: I2 /= np.sqrt(2.)
    return recursive_fdmt(I2,curMax,depth-1)


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
