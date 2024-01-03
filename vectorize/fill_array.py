import numpy as np
from timeit import default_timer as timer
from numba import cuda, jit, float64, void, prange
from time import perf_counter_ns

# Increment each element in an array in place

# This function will run on a CPU
def FillArrayWithoutGPU(a):
    for k in range(10000000):
        a[k]+=1

# This function will run on a CPU, JIT Python
@jit(void(float64[:]), nopython=True)
def FillArrayWithJIT(a):
    for k in range(10000000):
        a[k]+=1

# This function will run on multiple CPUs, JIT Python, parallel
@jit(void(float64[:]), nopython=True, parallel=True)
def FillArrayWithJITParallel(a):
    for k in prange(10000000):
        a[k]+=1

# This function will run on a GPU
# @jit(nopython=True, target_backend='cuda')
@jit(void(float64[:]), nopython=True, target_backend='cuda')
def FillArrayWithGPU(a):
    for k in range(10000000):
        a[k]+=1

def main():
    a = np.ones(10000000, dtype = np.float64)

    print("Using one CPU")
    start = timer()
    FillArrayWithoutGPU(a)
    print("On a CPU, " , timer()-start, " seconds")

    print("On a CPU using JIT")
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithJIT(a)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 50
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithJIT(a)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    # CUDA
    print("Using CUDA")
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        FillArrayWithGPU(a)
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 50
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        FillArrayWithGPU(a)
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 
 
    # Use parallel
    print("Parallel Multiple CPUs")
    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithJITParallel(a)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    times_to_run = 50
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithJITParallel(a)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

if __name__ == "__main__" :
    main()
