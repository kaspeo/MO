import numpy as np

A = np.array([[2, 0, 1, 2],[0, 2, 3, 1],[1,3,0,0],[2,1,0,0]])
B = np.array([[14],[6],[18],[7]])
A_t=np.linalg.inv(A)
Y=A_t.T@B
print(Y)

C = np.array([[2, 0, 2],[0, 2, 1],[2,1,0]])
D = np.array([[14],[6],[7]])
C_t=np.linalg.inv(C)
X=C_t.T@D
print(X)