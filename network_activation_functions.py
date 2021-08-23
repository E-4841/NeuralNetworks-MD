import numpy as np
#import theano
from keras import backend as T

# This module contains several activation functions to be used with
# the theano network class. Each activation takes a real number as
# input and is arbitrarily broadcastable.

def hyperbolic_tangent(x):
    return 1.7159 * T.tanh((2*x)/3)

def hyperbolic_tangent_with_linear_twist(x):
    return 1.7159 * T.tanh((2*x)/3) + 0.001*x

def sigmoid(x):
    return T.nnet.sigmoid(x)

def identity(x):
    return x

def rectilinear(x):
    return theano.ifelse.ifelse(T.lt(x,0),0.1*x,x)
# returns 0.1x for x<0 and x for x>0



#*********************** DERIVATIVES ***********************#

def hyperbolic_tangent_derivative(x):
    return ( 1 - np.tanh( (2*x)/3 )**2 )*1.7159*2/3

def hyperbolic_tangent_with_linear_twist_derivative(x):
    return ( 1 - np.tanh( (2*x)/3 )**2 )*1.7159*2/3 + 0.001

def sigmoid_derivative(x):
    s = 1/( 1 + np.exp(-x) )
    return s * (1 - s)

def identity_derivative(x):
    array_shape = x.shape
    return np.ones(array_shape)

#def rectilinear_derivative(x):
#    return 