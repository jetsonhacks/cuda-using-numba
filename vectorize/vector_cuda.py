import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda
from time import perf_counter_ns

@vectorize(["float32(float32, float32)"], target = "cuda")
def multiply_vector_elements_cuda(a,b):
    return a*b

def main():
    N = 64000000

    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)
    C = np.zeros(N, dtype = np.float32)

    times_to_run = 1
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        C = multiply_vector_elements_cuda(A, B)
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
        C = multiply_vector_elements_cuda(A, B)
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6

    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    print("C[:6] = " + str(C[:6]))
    print("C[:-6] = " + str(C[:-6]))

    
if __name__ == '__main__' :
    main()