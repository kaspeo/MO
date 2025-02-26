import numpy as np


def test(a,b):
    if a == b:
        print("True")
    else:
        print("False")

if __name__ == '__main__':
    vector=np.array([1,9,0,9,6,1])
    print(np.linalg.norm(vector,ord=1))
    print(np.linalg.norm(vector, ord=2))
    print(np.linalg.norm(vector, ord=4))
    print(np.linalg.norm(vector, ord=np.inf),"\n")

    a = np.matrix('2 6; 0 2')
    b = np.matrix('2 0; 2 5')
    print(a@b,"\n")
    print(np.multiply(a,b),"\n")
    print(np.kron(a,b))
    AB=(np.dot(a,b))
    a_inv = np.linalg.inv(a)
    b_inv = np.linalg.inv(b)
    AB_inv = np.linalg.inv(AB)
    ab_inv=a_inv*b_inv
    #test(AB_inv,ab_inv)
    print("---------------")
    print(AB_inv)
    print(ab_inv)
    print(np.linalg.det(a))
    print(np.linalg.det(b))

    V = np.matrix('2 6 2; 0 2 0; 2 5 2')
    print(np.linalg.det(V))
