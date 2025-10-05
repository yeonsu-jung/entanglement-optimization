import numpy as np
from matplotlib import pyplot as plt

def example1():
    K1 = 1.e5
    dist = np.logspace(-7,-3,100)

    v = np.exp(K1*dist)
    print(v)

    plt.plot(dist,v,'o-')
    plt.show()

def example2():
    K1 = 1.e5
    dist = np.logspace(-5,-3,100)*(-1.)
    
    v = np.exp(-K1*dist)
    w = np.exp(K1*dist)
    
    f = (-2 * v * np.log(v + 1)) / (K1 * (v + 1))
    g = 2*K1*dist/(K1*(1+w))
    error = np.abs((f-g)**2).mean()/np.abs(f).mean()
    print(error)
    
    plt.plot(dist,v,'o-',label='v')
    plt.show()
    
    plt.plot(dist,f,'.-',label='f')
    plt.plot(dist,g,'.-',label='g',alpha=0.5)
    plt.title(f"Error: {error}")
    plt.legend()
    plt.savefig("/Users/yeonsu/Figures/collision_force_approximation.png")
    
def example3():
    dist = np.logspace(-3,0,100)*(-1)
    K1 = 3.e5
    
    exponent = -K1*dist
    threshold = 709
    
    # v = np.zeros_like(exponent)
    # v = np.exp(exponent[exponent<threshold])
    # print(v)
    
    iter = 0
    f = []
    for d in dist:
        if -K1*d < threshold:
            w = np.exp(K1*d)
            f.append(2*K1*d/(K1*(1+w)))
        else:
            v = np.exp(-K1*d)
            f.append(-2*v*np.log(v+1)/(K1*(v+1)))
    
    print(f)
    
    plt.plot(dist,np.array(f))
    plt.show()

def main():
    example3()
    
if __name__ == '__main__':
    main()