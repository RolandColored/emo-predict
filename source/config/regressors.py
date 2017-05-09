from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


regressors_dict = {
    'Baseline': DummyRegressor(),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'ExtraTrees': ExtraTreesRegressor(),
    'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor()),
    'KNeighbours': KNeighborsRegressor(),
    'SVRLinear': MultiOutputRegressor(SVR(kernel='linear')),
    'SVRRbf': MultiOutputRegressor(SVR(kernel='rbf')),
    'SVRPoly': MultiOutputRegressor(SVR(kernel='poly')),
    'SVRSigmoid': MultiOutputRegressor(SVR(kernel='sigmoid')),
}
