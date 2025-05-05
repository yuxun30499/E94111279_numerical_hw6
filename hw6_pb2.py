import numpy as np

def inverse_via_gaussian_elimination(A):
    n = A.shape[0]
    I = np.eye(n)
    AI = np.hstack((A.astype(float), I))

    for i in range(n):
        # Pivot if necessary
        max_row = np.argmax(np.abs(AI[i:, i])) + i
        if max_row != i:
            AI[[i, max_row]] = AI[[max_row, i]]
        
        # Normalize pivot row
        AI[i] = AI[i] / AI[i, i]

        # Eliminate other rows
        for j in range(n):
            if j != i:
                AI[j] = AI[j] - AI[j, i] * AI[i]

    return AI[:, n:]  # Right half is inverse

# 題目中的矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
])

A_inv = inverse_via_gaussian_elimination(A)
print("A inverse =")
print(np.round(A_inv, 6))
