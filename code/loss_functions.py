import math
import numpy as np


def CrossEntropy(yHat, y):
    pass


def Hinge(yHat, y):
    pass


def KLDivergence(yHat, y):
    pass


def L1(yHat, y):
    pass


def L2(yHat, y):
    pass


def MLE(yHat, y):
    pass


def MSE(yHat, y):
    return np.sum((yHat - y)**2) / 2.0

def MSE_prime(yHat, y):
    return yHat - y
