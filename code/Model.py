from scipy import stats as __stats
import numpy as __np

from sklearn import linear_model as __linear_model

def LinearRegress(X, y):
    """
    """
    return __np.reshape(__np.linalg.inv(X.T@X)@X.T@__np.reshape(y, [-1,1]), [-1])

def log_likelihood_regression(mean, cov, y):
    """
    """
    return __stats.multivariate_normal(mean = mean, cov = cov).logpdf(y)/y.shape[0]


def __sigmoid(x):
    return __np.exp(-__np.logaddexp(0, -x))

def LogisticRegress(X, y):
    """
    """
    if y.all() == True:
        return LinearRegress(X, y + 5)
    if y.any() == False:
        return LinearRegress(X, y - 5)

    model = __linear_model.LogisticRegression(C = 10000000000)
    model.fit(X, y)
    return model.coef_[0]

def log_likelihood_logistic(mean, cov, y):
    """
    """
    epsilon = 0.00000000000001
    answers = __sigmoid(mean)
    answers[__np.where(answers < epsilon)] = epsilon
    return __np.mean(y*__np.log(answers) + (1-y)*__np.log(answers))

