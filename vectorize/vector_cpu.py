import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda
from time import perf_counter_ns

# valid targets are cpu (default), cuda, parallel
# This is similar to a numpy ufunc

@vectorize(["float32(float32, float32)"], target='cpu')
def multiply_vector_elements(a,b):
    return a*b

def main():
    N = 64_000_000

    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)
    C = np.zeros(N, dtype = np.float32)

    start = timer()
    C = multiply_vector_elements(A, B)
    vector_multiply_time = timer() - start
    print("This multiplication took %f seconds" % vector_multiply_time)
    print("C[:6] = " + str(C[:6]))
    print("C[:-6] = " + str(C[:-6]))

    times_to_run = 100
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        C = multiply_vector_elements(A, B)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6

    print(f"Elapsed time: {timing.mean():.3f} +- {timing.std():.3f} ms") 

    print("C[:6] = " + str(C[:6]))
    print("C[:-6] = " + str(C[:-6]))

    
if __name__ == '__main__' :
    main()