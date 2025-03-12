import numpy as np
import matplotlib.pyplot as plt

def predict(X,a,b):
    return a*X+b
def loss(X,Y,a,b):
    diff=predict(X,a,b)-Y
    return np.linalg.norm(diff, ord=2)

def train(X,Y,iterations,tr):
    a=0
    b=0
    for i in range(iterations):
        current_loss=loss(X,Y,a,b)
        if loss(X,Y,a+tr,b)<current_loss:
            a+=tr
        elif loss(X,Y,a-tr,b)<current_loss:
            a-=tr
        elif loss(X,Y,a,b+tr)<current_loss:
            b+=tr
        elif loss(X,Y,a,b-tr)<current_loss:
            b-=tr
    return a,b

if __name__ == '__main__':
    X,Y =np.loadtxt("dane.txt",skiprows=1, unpack=True)
    a,b=train(X,Y, 1000,0.01)
    plt.xlabel("natezenie")
    plt.ylabel("napiecie")
    Y_predict=predict(X,a,b)
    plt.plot(X,Y,'bo')
    plt.plot(X,Y_predict,'r')
    plt.show()

