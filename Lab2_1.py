import numpy as np
def gendata(a,b,num_points):
    x_val=np.random.uniform(0,10,num_points)
    y_val = a * x_val + b
    y_val+=np.random.normal(scale=2,size=num_points)
    with open("dane.txt","w") as file:
        file.write("Napiecie        natezenie\n")
        for x,y in zip(x_val,y_val):
            file.write(f"{x} {y} \n")
    print("Zapisano plik")

if __name__ == '__main__':
    a = 3.13
    b = 1.41
    num_points = 100
    gendata(a,b,num_points)