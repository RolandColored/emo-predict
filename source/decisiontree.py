from sklearn.tree import DecisionTreeRegressor


class DecisionTree:

    model = None

    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=0)

