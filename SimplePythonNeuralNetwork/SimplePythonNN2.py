import numpy as np

# Define inputs and training outputs
# X = np.array([(1/np.sqrt(2), 1/np.sqrt(2), 0),
#               (0, 0, 0),
#               (0, 0, 1),
#               (0, 1, 0),
#               (0, 0, 1),
#               (1, 0, 0)])

# y = np.array([[0, 0, 1, 0, 1, 0]]).T


X = np.array( [(0, 0, 1)])

y = np.array([[1]]).T

syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1



def nonlin(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


for i in range(1):
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y - l2
    l2_delta = l2_error*nonlin(l2,True)


    l1_error = np.dot(l2_delta,syn1.T)
    l1_delta = l1_error*nonlin(l1,True)

    syn0 += np.dot(l0.T,l1_delta)
    syn1 += np.dot(l1.T,l2_delta)
