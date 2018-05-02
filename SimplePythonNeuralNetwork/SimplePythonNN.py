import numpy as np

# Define inputs and training outputs
X = np.array([(1/np.sqrt(2), 1/np.sqrt(2), 0),
              (0, 0, 0),
              (0, 0, 1),
              (0, 1, 0),
              (0, 0, 1),
              (1, 0, 0)])

y = np.array([[0, 0, 1, 0, 1, 0]]).T

syn0 = 2*np.random.random((3,1))-1


def nonlin(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


for i in range(15000):
    l0 = X
    l1 = nonlin(np.dot(X,syn0))

    l1_error = y - l1
    l1_delta = l1_error*nonlin(l1,True)

    syn0 += np.dot(l0.T,l1_delta)
