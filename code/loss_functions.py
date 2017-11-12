import math
import numpy as np

# using the common convention:
#	yHat is the prediction while is the truth (true label)

def CrossEntropy(yHat, y):
    pass


def Hinge(yHat, y):
    pass


def KLDivergence(yHat, y):
    pass


def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))


def L2(yHat, y):
	# same as MSE but without mean
    return np.sum((yHat - y)**2)


def MLE(yHat, y):
    pass


def MSE(yHat, y):
    return np.sum((yHat - y)**2) / 2.0

def MSE_prime(yHat, y):
    return yHat - y
