import scipy

# def pole(X):
#     x1,x2=X
#     return -x1*x2
#
# def obwod(X):
#     return 2*(X[0]+X[1])-10
#
# X0=(1., 4.)
# ograniczenie=[{'type':'eq','fun':obwod}]
# wynik=scipy.optimize.minimize(pole,X0,constraints=ograniczenie)
# if wynik.success:
#     x1,x2=wynik.x
#     print("Długości boków to:",x1,"i",x2)
# else:
#     print("Urzadzenie ma defekt")

def obj(X):
    x1,x2,x3=X
    return x1*x2*x3-300

def pow(X):
    x1, x2, x3 = X
    return 2*(x1*x2+x1*x3+x2*x3)

def pol(X):
    x1,x2,x3=X
    return x1*x2-50.0


X0=(1., 2., 3.)
# ograniczenie=[{'type':'eq','fun':obj}]
ograniczenie=[{'type':'eq','fun':obj},{'type':'ineq','fun':pol}]
wynik=scipy.optimize.minimize(pow,X0,method='SLSQP',constraints=ograniczenie)
if wynik.success:
    x1,x2,x3=wynik.x
    print("Długości boków to:",x1,",",x2,",",x3)
else:
    print("Urzadzenie ma defekt")