from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


regressor_list = [
    Ridge(),
    KernelRidge(kernel='linear'),
    KernelRidge(kernel='rbf'),
    KernelRidge(kernel='poly'),
    KernelRidge(kernel='sigmoid'),
    MultiOutputRegressor(BayesianRidge()),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    ExtraTreesRegressor(),
    MultiOutputRegressor(GradientBoostingRegressor()),
    KNeighborsRegressor(),
    MultiTaskLasso(),
    MultiTaskElasticNet(),
    MultiOutputRegressor(NuSVR(kernel='linear')),
    MultiOutputRegressor(NuSVR(kernel='rbf')),
    MultiOutputRegressor(NuSVR(kernel='poly')),
    MultiOutputRegressor(NuSVR(kernel='sigmoid')),
]


# helper function
def get_regressor_name(regressor_obj):
    if isinstance(regressor_obj, KernelRidge):
        regressor_name = regressor_obj.__class__.__name__ + regressor_obj.kernel
    elif isinstance(regressor_obj, MultiOutputRegressor):
        regressor_name = regressor_obj.estimator.__class__.__name__
        if isinstance(regressor_obj.estimator, NuSVR) or isinstance(regressor_obj.estimator, SVR):
            regressor_name += regressor_obj.estimator.kernel
    else:
        regressor_name = regressor_obj.__class__.__name__
    return regressor_name


def get_n_jobs(regressor_name, max_n_jobs):
    if "Ridge" in regressor_name:
        return 1
    return max_n_jobs
