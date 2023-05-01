import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def eta(x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)


def entropy_loss(y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss
def categorical_cross_entropy_loss(y, yhat):
    nsample = len(y)
    yhat = eta(yhat) # clips value to avoid NaNs in log
    loss = -1/nsample * np.sum(y * np.log(yhat))
    return loss

a = np.zeros([3,3])
a[0,0] = 1
a[1,1] = 1
a[2,2] = 1
b = np.zeros([3,3])
b[0,0] = 1
b[1,2] = 1
b[2,1] = 1

# print(entropy_loss(a,b))
y = np.argmax(a, axis=1)
yhat = np.argmax(b, axis=1)
print(y)
print(yhat)