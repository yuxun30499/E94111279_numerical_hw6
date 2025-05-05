import numpy as np

def crout_tridiagonal(A, b):
    n = len(b)
    L = np.zeros((n, n))
    U = np.identity(n)

    # 分解 A = LU
    L[0, 0] = A[0, 0]
    for i in range(n - 1):
        U[i, i + 1] = A[i, i + 1] / L[i, i]
        L[i + 1, i] = A[i + 1, i]
        L[i + 1, i + 1] = A[i + 1, i + 1] - L[i + 1, i] * U[i, i + 1]

    # 前代：Ly = b
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - L[i, i - 1] * y[i - 1]) / L[i, i]

    # 回代：Ux = y
    x = np.zeros(n)
    x[-1] = y[-1]
    for i in reversed(range(n - 1)):
        x[i] = y[i] - U[i, i + 1] * x[i + 1]

    return x

# 題目中的矩陣 A 和 b
A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
])
b = np.array([2, 3, 4, 1])

x = crout_tridiagonal(A, b)
print("解為：")
for i, xi in enumerate(x, 1):
    print(f"x{i} = {xi:.6f}")
