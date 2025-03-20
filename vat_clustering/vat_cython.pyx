import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

def vat_cython(cnp.ndarray[cnp.float64_t, ndim=2] R):
    """
    Optimized VAT using Cython for maximum speed.
    """
    cdef int N = R.shape[0]
    cdef int i, j, j_star, min_idx
    cdef double min_val

    # Allocate C arrays
    cdef int* J = <int*> malloc(N * sizeof(int))
    cdef int* I = <int*> malloc(N * sizeof(int))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] RV = np.zeros((N, N), dtype=np.float64)

    if not J or not I:
        raise MemoryError("Memory allocation failed.")

    # Initialize J and I
    for i in range(N):
        J[i] = i
    I[0] = np.argmax(np.sum(R, axis=1))
    
    # Remove the first selected index from J
    for i in range(N):
        if J[i] == I[0]:
            J[i] = J[N-1]
            break

    # VAT Algorithm Execution
    for i in range(1, N):
        min_val = float("inf")
        for j in range(N - i):
            for j_star in range(i):
                if R[J[j], I[j_star]] < min_val:
                    min_val = R[J[j], I[j_star]]
                    min_idx = j
        
        I[i] = J[min_idx]
        J[min_idx] = J[N - i - 1]

    # Construct Reordered Dissimilarity Matrix
    for i in range(N):
        for j in range(N):
            RV[i, j] = R[I[i], I[j]]

    # Convert I (C array) to a NumPy array before returning
    I_numpy = np.array([I[k] for k in range(N)], dtype=np.int32)

    # Free allocated memory
    free(J)
    free(I)

    return RV, I_numpy  # ✅ Return NumPy array instead of C pointer
