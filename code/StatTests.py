from scipy import stats as __stats
import numpy as __np

def independence(X):
    pass


def NormalTest(X, axis = 0):
    """
    return p-values
    """
    _, p_values = __stats.normaltest(X , axis = axis)
    return p_values

def UniformTest(X):
    """
    X подается в виде [l,n], где n --- количество признаков, l --- количество объектов
    return p_value
    """
    List_d = []
    
    for i in range(X.shape[1]):
        d, _ = __np.histogram(X[:,i], bins = X.shape[0]//10 + 1)
        List_d.append(d)
    _, p_value = __stats.chisquare(__np.array(List_d), axis = 1)
    
    return p_value

def BernoulliTest(X):
    pass