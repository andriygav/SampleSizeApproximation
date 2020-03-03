import numpy as _np
import pandas as _pd


from sklearn.datasets import load_boston as _load_boston
from sklearn.datasets import load_iris as _load_iris
from sklearn.datasets import load_diabetes as _load_diabetes
from sklearn.datasets import load_wine as _load_wine





def LoadData(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data", 
             sep=',', names=["motor", "screw", "pgain", "vgain", "answer"]):
    """
    return X, y
    """
    data = _pd.read_csv(url, sep=sep, names=names)
    y = data['answer'].values
    del data['answer']
    X = data.values
    return X, y

def DataLoader(name = "servo", is_binary_answ = True):
    """
    1. boston reg
    2. servo reg
    3. wine class
    4. diabetes reg
    5. iris class
    6. forestfires reg
    7.
    8.
    9.
    10.
    11.
    12. 
    
    """
   
    if name == 'servo':
        X, y = LoadData(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data", 
                         sep=',', names=["motor", "screw", "pgain", "vgain", "answer"])
        X[_np.where(X == 'A')] = 1
        X[_np.where(X == 'B')] = 2
        X[_np.where(X == 'C')] = 3
        X[_np.where(X == 'D')] = 4
        X[_np.where(X == 'E')] = 5
        X = _np.array(X, dtype = _np.float64)
            
    if name == 'boston':
        dataset = _load_boston()
        X = dataset.data
        y = dataset.target
        
    if name == 'wine':
        dataset = _load_wine()
        X = dataset.data
        y = dataset.target
        X = _np.array(X, dtype = _np.float64)
        if is_binary_answ:
            X = _np.delete(X, _np.where(y == 2), axis = 0)
            y = _np.delete(y, _np.where(y == 2), axis = 0)
        
    if name == 'diabetes':
        dataset = _load_diabetes()
        X = dataset.data
        y = dataset.target
        X = _np.array(X, dtype = _np.float64)
        
    if name == 'iris':
        dataset = _load_iris()
        X = dataset.data
        y = dataset.target
        X = _np.array(X, dtype = _np.float64)
        if is_binary_answ:
            X = _np.delete(X, _np.where(y == 2), axis = 0)
            y = _np.delete(y, _np.where(y == 2), axis = 0)
            
    if name == 'forestfires':
        X, y = LoadData(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv", 
                        sep=',', names=["X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain","answer"])
        X = _np.delete(X, 0, axis = 0)
        y = _np.delete(y, 0, axis = 0)
        
        list_of_month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        list_of_day = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

        for i, mon in enumerate(list_of_month):
            X[_np.where(X == mon)] = i
        for i, mon in enumerate(list_of_day):
            X[_np.where(X == mon)] = i

        X = _np.array(X, dtype = _np.float64)
        y = _np.array(y, dtype = _np.float64)

    return X, y
    