import numpy as np

def gaussian_elimination_with_pivoting(A, b):
    n = len(b)
    # 增廣矩陣
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    for i in range(n):
        # Pivoting: 找最大行
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]  # swap rows

        # Elimination
        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]

    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
    return x

# 題目中的 A, b
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
])

b = np.array([1.12, 3.44, 2.15, 4.16])

x = gaussian_elimination_with_pivoting(A, b)
for i, val in enumerate(x, 1):
    print(f"x{i} = {val:.6f}")
