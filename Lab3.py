import numpy as np

A = np.array([[1, 2, 0],[2, 0, 2]])
print(A)
A_t= A.transpose()
A_tA=np.dot(A_t,A)
AA_t=np.dot(A,A_t)
S,U=np.linalg.eig(AA_t)
print("Wartosc wlasna",S,"\n")
print("Wektro wlasny",U,"\n")
sigma=np.sqrt(S)
sigma_r=np.diag(sigma)
print("sigma_r:\n",sigma_r,"\n")
u1=U[:,0]
u2=U[:,1]
print("u1:",u1)
print("u2",u2)
v1=1./sigma[0]*A_t@u1
v2=1./sigma[1]*A_t@u2
V_r=np.array([v1,v2]).T
Acalc=U@sigma_r@V_r.T
print("\nWynik koncowy:\n",Acalc)
v3=np.cross(v1,v2)
V=np.array([v1,v2,v3]).T
zero_column=np.zeros((sigma_r.shape[0],1))
Sigma=np.hstack((sigma_r,zero_column))
print(Sigma)
Acalc2=U@Sigma@V.T
print("\nWynik koncowy2:\n",Acalc2)
SigmaPlus_r=np.diag(1.0/sigma)
zeros_row=np.zeros((1,SigmaPlus_r.shape[0]))
SigmaPlus=np.vstack((SigmaPlus_r,zeros_row))
APlus=V@SigmaPlus@U.T
print("\nAPlus:\n",APlus)
print("\n",A@APlus)
print("/////////////////////////////////////////////")

U_np,S_np,V_np=np.linalg.svd(A)
print("U_np:",U_np)
print("S_np:",S_np)
print("V_np:",V_np)
