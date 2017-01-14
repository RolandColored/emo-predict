from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor


regressor_list = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    MultiTaskLasso(),
    MultiTaskElasticNet(),
    MultiOutputRegressor(BayesianRidge()),
    MultiOutputRegressor(GradientBoostingRegressor()),
    MultiOutputRegressor(NuSVR(kernel='rbf')),
    MultiOutputRegressor(NuSVR(kernel='poly')),
    MultiOutputRegressor(NuSVR(kernel='sigmoid')),
    ExtraTreesRegressor(),
    MultiOutputRegressor(AdaBoostRegressor()),
]
