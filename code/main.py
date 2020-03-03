import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy as sp

import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import metrics

from scipy import stats

from tqdm import tqdm

from sklearn.datasets import load_boston

from sklearn.datasets import load_iris

from sklearn.datasets import load_diabetes

from sklearn.datasets import load_wine

import pickle

#Mylib
import Ordering as OD
import StatTests as st
import Model as md


ordering, SampleStep, List_mean, List_std, X_train, y_train, X_test, y_test = pickle.load(open("saved_graph/diabets5000.p", "rb"))

x = SampleStep
y = 1-(np.linspace(0, ordering.shape[0] - 2, ordering.shape[0]-1, dtype = np.int64) - (ordering.shape[0] - 2))

xgrid, ygrid = np.meshgrid(x, y)
zgrid = np.array(List_mean)

fig = plt.figure()
axes = Axes3D(fig)
axes.plot_wireframe(xgrid, ygrid, zgrid, rstride=3, cstride=20, color ='black')
axes.plot_surface(xgrid, ygrid, zgrid, alpha = 0.1, color ='black')

axes.set_xlabel("number of objects", labelpad=30)
axes.set_ylabel("number of features", labelpad=20)
axes.set_zlabel("log-likelihood", labelpad=20)

plt.show()


