#!/usr/bin/env python

import sys
import numpy as np
from time import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

fmin    = 400.
fmax    = 800.
nchan   = 4096
maxDT   = 8192
T       = 61440
fs, df  = np.linspace(fmin,fmax,nchan,endpoint=False,retstep=True)

I       = None  # input
A       = None  # working area (GPU) 
B       = None  # loading stage (GPU)
ch_gulp = 64    # channels loaded at once 
plan    = None  # GPU execution plan

negative_DMs = False
cor = df/32

class ExecutionPlan:
    """
     A plan is generated once to facilitate above input-output specs. 
     It can then be reused to process different inputs.
    """ 
    def __init__(self):
        self.funcs = []
        self.args  = []
    
    def execute(self):
        cuda.start_profiler()
        timestamp = time()
        for i,(f,a) in enumerate(zip(self.funcs,self.args)):
            f(*a)
        gpu_transpose.prepared_call(grid2,block2,A)
        gpu_get_sigma.prepared_call(grid3,block3,A)
        cuda.Context.synchronize()
        read_results()
        cuda.stop_profiler()
        print "Time for DM tranform, transpose, sigma calculate, get max sigma: %.3f s" % (time()-timestamp)

def fdmt(Image):
    global I
    I = Image
    assert I.shape[0] == nchan, 'Expected %i channels.' % nchan
    assert I.shape[1] == T, 'Expected %i samples.' % T
    if A is None: allocateGPU()
    if plan is None: generate_plan()
    plan.execute()


def allocateGPU():
    global A,B
    bw = fmax - fmin
    A_memReq = maxDT + subDT(fmin,bw/2) + subDT(fmin+bw/2,bw/2)
    A = cuda.mem_alloc(A_memReq * T * 4) 
    B = cuda.mem_alloc(ch_gulp * T * 4)


def subDT(f,dF=df):
    "Get needed DT of subband to yield maxDT over entire band"
    loc = f**-2 - (f+dF)**-2
    glo = fmin**-2 - fmax**-2
    return np.ceil((maxDT-1)*loc/glo).astype(int)+1


def generate_plan():
    global plan
    plan = ExecutionPlan()
    plan.funcs.append(cuda.memcpy_htod)
    plan.args.append([B,I[:ch_gulp]])
    for i in xrange(nchan/2):
        add_chs(i)
        async = (i+1)%(ch_gulp/2) == 0 and i < nchan/2 - 2
        for j in xrange(1,int(np.log2(nchan))):
            if (i+1)%(2**j) == 0: 
                join(fs[2*i+1]+df,j,async)
        if async: load_chs(2*i+2)
    

class MemoryManager:
    top    = np.int32(maxDT)
    bottom = np.int32(maxDT)
mem = MemoryManager()


def add_chs(i):
    "Initializes a pair of channels and then joins them"
    DT01,_,DT02,C01,C12 = get_joining_params(fs[2*i]+2*df,df)
    Ai = mem.bottom - 2*DT01
    Aj = mem.bottom - DT01
    Ak = mem.top
    Bi = np.int32((2*i)%ch_gulp)
    Bi = np.int32(0)
    mem.top += DT02
    if negative_DMs: DT02 *= -1
    plan.funcs.append(gpu_add_chs.prepared_call)
    plan.args.append([grid,block,A,B,Bi,Ai,Aj,Ak,DT01,DT02,C01,C12])

def get_joining_params(f2,dF):
    "Produces joining info based on top freq. and pre-join BW"
    f0 = f2 - 2*dF
    f1 = f2 - dF  
    DT01 = np.int32(subDT(f0,dF))
    DT12 = np.int32(subDT(f1,dF))
    DT02 = np.int32(subDT(f0,2*dF))
    C01 = np.float64(((f1-cor)**-2-f0**-2)/(f2**-2-f0**-2))
    C12 = np.float64(((f1+cor)**-2-f0**-2)/(f2**-2-f0**-2))
    return DT01, DT12, DT02, C01, C12


def join(f2,d,async):
    "Join subtransforms given top freq. and current depth (past joins)"
    DT01,DT12,DT02,C01,C12 = get_joining_params(f2,df*2**d)
    if d%2 == 0:
        Aj = mem.bottom 
        Ai = Aj + DT12 
        Ak = mem.top 
        mem.bottom += DT01 + DT12
        mem.top += DT02
    else:
        Ai = mem.top - DT12 - DT01
        Aj = Ai + DT01
        Ak = mem.bottom - DT02
        mem.bottom -= DT02
        mem.top -= DT01 + DT12
    if negative_DMs: DT02 *= -1
    if async: 
        plan.funcs.append(gpu_join.prepared_async_call)
        plan.args.append([grid,block,strm1,A,Ai,Aj,Ak,DT02,C01,C12])
    else:
        plan.funcs.append(gpu_join.prepared_call)
        plan.args.append([grid,block,A,Ai,Aj,Ak,DT02,C01,C12])


def load_chs(f0_i):
    "Loads the next chunk of channels to GPU"
    plan.funcs.append(cuda.memcpy_htod_async)
    plan.args.append([B,I[f0_i:f0_i+ch_gulp],strm2])


def read_results():
    out = np.empty(maxDT,np.float32)
    cuda.memcpy_dtoh(out,A)
    print "Max sigma:", out.max()
    
    # if dedispersed results are desired...
    #out = np.empty((T-maxDT,maxDT),np.float32)
    #cuda.memcpy_dtoh(out,int(A)+maxDT*T*4) #dedispersed zone
    #return out
 
#####################################################################
############################ GPU STUFF ##############################
#####################################################################

TILE_DIM = 64
BLOCK_ROWS = 16

defines = "#define T %i\n" % T +\
          "#define MAXDT %i\n" % maxDT +\
          "#define TILE_DIM %i\n" % TILE_DIM +\
          "#define BLOCK_ROWS %i\n" % BLOCK_ROWS

mod = SourceModule(defines+"""

__global__ void add_channels(
    float *A,
    float *B,
    int Bi,
    int Ai,
    int Aj,
    int Ak,
    int iDT,
    int kDT,
    float C01,
    float C12
){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // Initialize
    int i_sum = 0;
    int j_sum = 0;
    for (int i_dT = 0; i_dT < iDT; i_dT++){
        int offset = (kDT > 0) ? tid-i_dT : tid+i_dT;
        if (offset >= 0 && offset < T) {
            i_sum += __ldg(&B[Bi*T+offset]);
            j_sum += __ldg(&B[(Bi+1)*T+offset]);
        }
        A[(Ai+i_dT)*T + tid] = i_sum;
        A[(Aj+i_dT)*T + tid] = j_sum;
    }
    // Join
    for (int i_dT = 0; i_dT < abs(kDT); i_dT++){
        int dT_mid01 = int(rint(i_dT * C01));
        int dT_mid12 = int(rint(i_dT * C12));
        int dT_rest = i_dT - dT_mid12;
        int src_i = (Ai+dT_mid01)*T + tid;
        int dst_k = (Ak+i_dT)*T + tid;
        if ((tid < dT_mid12 && kDT > 0) || (tid+dT_mid12 >= T && kDT < 0)) {
            A[dst_k] = __ldg(&A[src_i]);
        } else {
            int src_j = (kDT > 0) ? (Aj+dT_rest)*T + tid-dT_mid12 :
                                    (Aj+dT_rest)*T + tid+dT_mid12 ;
            A[dst_k] = __ldg(&A[src_i]) + __ldg(&A[src_j]); 
        }
    }
}

__global__ void join_transforms(
    float *A,
    int Ai,
    int Aj,
    int Ak,
    int kDT,
    float C01,
    float C12
){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i_dT = 0; i_dT < abs(kDT); i_dT++){
        int dT_mid01 = int(rint(i_dT * C01));
        int dT_mid12 = int(rint(i_dT * C12));
        int dT_rest = i_dT - dT_mid12;
        int src_i = (Ai+dT_mid01)*T + tid;
        int dst_k = (Ak+i_dT)*T + tid;
        if ((tid < dT_mid12 && kDT > 0) || (tid+dT_mid12 >= T && kDT < 0)) {
            A[dst_k] = __ldg(&A[src_i]);
        } else {
            int src_j = (kDT > 0) ? (Aj+dT_rest)*T + tid-dT_mid12 :
                                    (Aj+dT_rest)*T + tid+dT_mid12 ;
            A[dst_k] = __ldg(&A[src_i]) + __ldg(&A[src_j]); 
        }
    }
}

__global__ void transposeDiagonal(float *A)
{
    __shared__ int tile[TILE_DIM][TILE_DIM+1];

    // diagonal reordering
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    int blockIdx_y = bid%gridDim.y;
    int blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;

    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x + MAXDT;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*T;

    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*MAXDT;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y+i][threadIdx.x] = A[index_in+i*T];
    }

    __syncthreads();
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        A[index_out + (i+T)*MAXDT] = tile[threadIdx.x][threadIdx.y+i];
    }
}

__global__ void getSigma(float *A){
    int tid = threadIdx.x + blockDim.x * blockIdx.x + MAXDT * T;
    int len = T-MAXDT;
    float cur, s, ss, mean, rms;
    for (int i = 0; i < len; i += 1){
        cur = A[tid+i*MAXDT];
        s  += cur;
        ss += cur*cur;
    }
    mean = s/float(len);
    rms  = sqrt(ss/float(len) - mean*mean);
    float maxSig = 0;
    for (int i = 0; i < len; i += 1){
        cur = (A[tid+i*MAXDT]-mean)/rms;
        maxSig = (cur > maxSig) ? cur : maxSig;
        A[tid+i*MAXDT] = cur; 
    }
    A[tid-T*MAXDT] = maxSig; 
}

""")

gpu_add_chs = mod.get_function('add_channels')
gpu_join = mod.get_function('join_transforms')
gpu_transpose = mod.get_function('transposeDiagonal')
gpu_get_sigma = mod.get_function('getSigma')

gpu_add_chs.prepare("PPIIIIIiff")
gpu_join.prepare("PIIIiff")
gpu_transpose.prepare("P")
gpu_get_sigma.prepare("P")

strm1 = cuda.Stream()  # Used for async computing
strm2 = cuda.Stream()  # Used for async data transfer

tpb = 1024
grid  = (T/tpb,1)
block = (tpb,1,1)

grid2  = ((T-maxDT)/TILE_DIM,maxDT/TILE_DIM)
block2 = (TILE_DIM,BLOCK_ROWS,1)

grid3  = (maxDT/tpb,1)
block3 = (tpb,1,1)

#####################################################################

if __name__ == '__main__':
    I = np.random.random_integers(0,2**16-1,(nchan,T)).astype(np.float32)
    #I = np.ones((nchan,T),np.float32)
    print "Made array..."
    fdmt(I)
