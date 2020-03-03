import numpy as __np
from tqdm import tqdm as __tqdm

from sklearn import linear_model as __linear_model


#For LinearRegression
def __find_next_feature_LinearRegression(X, y, Ranking_of_parameters, alpha_min, alpha_max):
    """
    Выполняет поиск alpha, при котором количество учитываемых признаков уменьшяется на единицу.
    Поиск осуществляется при помощи бинарного поиска (делением отрезка пополам).
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    Ranking_of_parametets - вектор упорядочености признаков с прерыдущей итерации,
            каждый элемент вектора указывает каким по порядку он удаляется при увеличении параметра alpha.
    alpha - скаляр типа float, указывает какое alpha используется,
            для текущего количества удаленных признаков
    alpha_max - скаляр типа float, указывает максимально возможное значения параметра alpha.
    
    Возвращаемые значения
    ==========
    new_indexe - скаляр типа int, индекс следующего удаляемого признака, при увеличении alpha
    """
    curent_number_of_features = __np.sum(Ranking_of_parameters < 0)
    curent_indexes = __np.where(Ranking_of_parameters >= 0)[0]
    
    count_of_features = [0, 0, 0]
    alphas = [alpha_min, (alpha_min + alpha_max)/2.0, alpha_max]
    for i in range(3):
        model = __linear_model.Lasso(alphas[i])
        model.fit(X, y)
        count_of_features[i] = __np.sum(__np.abs(model.coef_) > 0)
    while(1):
        if count_of_features[1] >= curent_number_of_features:
            alphas[0] = alphas[1]
            alphas[1] = (alphas[0] + alphas[2])/2.0
            count_of_features[0] = count_of_features[1]
            model = __linear_model.Lasso(alphas[1])
            model.fit(X, y)
            count_of_features[1] = __np.sum(__np.abs(model.coef_) > 0)
            
        elif count_of_features[1] < curent_number_of_features - 1:
            alphas[2] = alphas[1]
            alphas[1] = (alphas[0] + alphas[2])/2.0
            count_of_features[2] = count_of_features[1]
            model = __linear_model.Lasso(alphas[1])
            model.fit(X, y)
            count_of_features[1] = __np.sum(__np.abs(model.coef_) > 0)
            
        elif count_of_features[1] == curent_number_of_features - 1:
            model = __linear_model.Lasso(alphas[1])
            model.fit(X, y)
            new_indexes = __np.where(__np.abs(model.coef_) == 0)[0]
            for i in range(len(curent_indexes)):
                if (curent_indexes[i] != new_indexes[i]):
                    return new_indexes[i]
            return new_indexes[-1]
    pass

def __find_max_alpha_LinearRegression(X, y):
    """
    Выполняет поиск такого alpha, при котором ни один признаки не учитываются.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    
    Возвращаемые значения
    ==========
    alpha - скаляр типа float, значения параметра alpha, при котором ни один из признаков не учитывается
    """
    alpha = 1
    while(1):
        model = __linear_model.Lasso(alpha=alpha)
        model.fit(X, y)
        if __np.sum(__np.abs(model.coef_) > 0) == 0:
            return alpha
        alpha += 1
    pass

def __find_min_alpha_LinearRegression(X, y):
    """
    Выполняет поиск такого alpha, при котором все признаки учитываются.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    
    Возвращаемые значения
    ==========
    alpha - скаляр типа float, значения параметра alpha, при котором все признаки учитываются
    """
    alpha = 1
    while(1):
        model = __linear_model.Lasso(alpha=alpha)
        model.fit(X, y)
        if __np.sum(__np.abs(model.coef_) == 0) == 0:
            return alpha
        alpha = alpha /2.0
    pass

def __features_ordering_LinearRegression(X, y, print_progres = False):
    """
    Выполняет упорядочивания признаков, при помощи L1 регулярицаии, 
    использует стандартную функцию из библиотеки sklearn __linear_model.Lasso(alpha), 
    в которой при увеличении alpha количество признаков убывает.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    print_progres - переменная типа bool, указывающая нужно ли выводить прогресс работы
    """
    Ranking_of_parameters = -1*__np.ones(X.shape[1], dtype = __np.int64)
    alpha_min = __find_min_alpha_LinearRegression(X, y)
    alpha_max = __find_max_alpha_LinearRegression(X, y) + 1
    
    if print_progres:
        list_of_size = __tqdm(range(X.shape[1]))
    else:
        list_of_size = range(X.shape[1])
    
    for _ in list_of_size:
        next_item = __find_next_feature_LinearRegression(X, y, Ranking_of_parameters, alpha_min, alpha_max)
        Ranking_of_parameters[next_item] = __np.max(Ranking_of_parameters) + 1
    
    return Ranking_of_parameters

#For LogisticRegression

def __find_next_feature_LogisticRegression(X, y, Ranking_of_parameters, alpha_min, alpha_max):
    """
    Выполняет поиск alpha, при котором количество учитываемых признаков уменьшяется на единицу.
    Поиск осуществляется при помощи бинарного поиска (делением отрезка пополам).
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    Ranking_of_parametets - вектор упорядочености признаков с прерыдущей итерации,
            каждый элемент вектора указывает каким по порядку он удаляется при увеличении параметра alpha.
    alpha - скаляр типа float, указывает какое alpha используется,
            для текущего количества удаленных признаков
    alpha_max - скаляр типа float, указывает максимально возможное значения параметра alpha.
    
    Возвращаемые значения
    ==========
    new_indexe - скаляр типа int, индекс следующего удаляемого признака, при увеличении alpha
    """
    curent_number_of_features = __np.sum(Ranking_of_parameters < 0)
    curent_indexes = __np.where(Ranking_of_parameters >= 0)[0]
    
    count_of_features = [0, 0, 0]
    alphas = [alpha_min, (alpha_min + alpha_max)/2.0, alpha_max]
    for i in range(3):
        model = __linear_model.LogisticRegression(C=alphas[i], penalty='l1')
        model.fit(X, y)
        count_of_features[i] = __np.sum(__np.abs(model.coef_[0]) > 0)
        
    while(1):
        if count_of_features[1] >= curent_number_of_features:
            alphas[2] = alphas[1]
            alphas[1] = (alphas[0] + alphas[2])/2.0
            count_of_features[2] = count_of_features[1]
            model = __linear_model.LogisticRegression(C=alphas[1], penalty='l1')
            model.fit(X, y)
            count_of_features[1] = __np.sum(__np.abs(model.coef_[0]) > 0)
            
        elif count_of_features[1] < curent_number_of_features - 1:
            alphas[0] = alphas[1]
            alphas[1] = (alphas[0] + alphas[2])/2.0
            count_of_features[0] = count_of_features[1]
            model = __linear_model.LogisticRegression(C=alphas[1], penalty='l1')
            model.fit(X, y)
            count_of_features[1] = __np.sum(__np.abs(model.coef_[0]) > 0)
            
        elif count_of_features[1] == curent_number_of_features - 1:
            model = __linear_model.LogisticRegression(C=alphas[1], penalty='l1')
            model.fit(X, y)
            new_indexes = __np.where(__np.abs(model.coef_[0]) == 0)[0]
            for i in range(len(curent_indexes)):
                if (curent_indexes[i] != new_indexes[i]):
                    return new_indexes[i]
            return new_indexes[-1]
    pass

def __find_max_alpha_LogisticRegression(X, y):
    """
    Выполняет поиск такого alpha, при котором все признаки учитываются.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    
    Возвращаемые значения
    ==========
    alpha - скаляр типа float, значения параметра alpha, при котором все признаки учитываются
    """
    alpha = 1
    
    while(1):
        model = __linear_model.LogisticRegression(C=alpha, penalty='l1')
        model.fit(X, y)
        if __np.sum(__np.abs(model.coef_[0]) == 0) == 0:
            return alpha
        alpha += 1
    pass

def __find_min_alpha_LogisticRegression(X, y):
    """
    Выполняет поиск такого alpha, при котором ни один признаки не учитываются.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной регрессии типа numpy.array, размера [l]
    
    Возвращаемые значения
    ==========
    alpha - скаляр типа float, значения параметра alpha, при котором ни один из признаков не учитывается
    """
    alpha = 1
    
    while(1):
        model = __linear_model.LogisticRegression(C=alpha, penalty='l1')
        model.fit(X, y)
        if __np.sum(__np.abs(model.coef_[0]) > 0) == 0:
            return alpha
        alpha = alpha /2.0
    pass
    

def __features_ordering_LogisticRegression(X, y, print_progres = False):
    """
    Выполняет упорядочивания признаков, при помощи L1 регулярицаии, 
    использует стандартную функцию из библиотеки sklearn __linear_model.LogisticRegression(alpha), 
    в которой при увеличении alpha количество признаков возрастает.
    
    Параметры
    ==========
    X - матрица обьектов типа numpy.array, размера [l, n]
    y - матрица ответов задачи одномерной классификациии типа numpy.array, размера [l]
    print_progres - переменная типа bool, указывающая нужно ли выводить прогресс работы
    
    Возвращаемые значения
    ==========
    Ranking_of_parameters - вектор numpy.array типа int размера [n], 
            каждое значения указывает номер по порядку, удаления параметров при увеличения значения alpha.
    """
    Ranking_of_parameters = -1*__np.ones(X.shape[1], dtype = __np.int64)
    alpha_min = __find_min_alpha_LogisticRegression(X, y)
    alpha_max = __find_max_alpha_LogisticRegression(X, y) + 1
    
    if print_progres:
        list_of_size = __tqdm(range(X.shape[1]))
    else:
        list_of_size = range(X.shape[1])
    
    for _ in list_of_size:
        next_item = __find_next_feature_LogisticRegression(X, y, Ranking_of_parameters, alpha_min, alpha_max)
        Ranking_of_parameters[next_item] = __np.max(Ranking_of_parameters) + 1
    
    return Ranking_of_parameters

#Common version
def features_ordering(X, y, print_progres=False, linear_model = 'LinearRegression'):
    """
    """
    if linear_model == 'LinearRegression':
        return __features_ordering_LinearRegression(X, y, print_progres = print_progres)
    if linear_model == 'LogisticRegression':
        return __features_ordering_LogisticRegression(X, y, print_progres = print_progres)
    return __np.ones(X.shape[1], dtype = __np.int64)