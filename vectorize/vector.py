import numpy as np
from timeit import default_timer as timer

from time import perf_counter_ns

def MultiplyMyVectors(a,b,c):
    for i in range(a.size):
        c[i] = a[i] * b[i]

def main():
    N = 64000000

    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)
    C = np.zeros(N, dtype = np.float32)

    start = timer()
    MultiplyMyVectors(A, B, C)
    vector_multiply_time = timer() - start
    print("C[:6] = " + str(C[:6]))
    print("C[:-6] = " + str(C[:-6]))
    print("This multiplication took %f seconds" % vector_multiply_time)
    
if __name__ == '__main__' :
    main()