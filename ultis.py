"""   IMPORT LIBRARY   """
import numpy as np
import pickle
"""   DATA PRE-PROCESSING   """


class DataSet:

    def __init__(self, df, x, y, train=False, scaling=False, label_encode=None, dtype='auto'):
        self.data = df.values
        self.X = df.iloc[:, x].values
        self.y = df.iloc[:, y].values.ravel()
        self.scaled = False
        self.split = False
        self.encoded = False

        """   Encode Features   """
        if label_encode == "auto":
            print("Auto Encoding Selected")
            label_encode = []
            for idx, col in enumerate(df.columns):
                if df[col].dtype == np.object and [idx] != y:
                    label_encode.append(idx)

            if label_encode:
                self.encode(label_encode)

        elif type(label_encode) is list:
            self.encode(label_encode)

        """   Scale Features   """
        if scaling:
            from sklearn.preprocessing import StandardScaler
            self.scaled = True
            sc = StandardScaler()
            self.X = sc.fit_transform(self.X)

        """   Train Split Features   """
        if train:
            self.create_training_set(train)

        if dtype == np.int8:
            self.X = self.X.astype(dtype)
        elif dtype == np.float32:
            self.X = self.X.astype(dtype)

    """     FUNCTIONS       """

    def encode(self, labels):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        print("Encoding Now")
        self.encoded = True
        for lbl in labels:
            labelencoder_X = LabelEncoder()
            self.X[:, lbl] = labelencoder_X.fit_transform(self.X[:, lbl])
        OHE = OneHotEncoder(categorical_features=labels)
        self.X = OHE.fit_transform(self.X).toarray()

    def create_training_set(self, train_split):
        from sklearn.model_selection import train_test_split
        self.split = True
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=train_split)


"""   USEFUL COMMANDS   """


class Models:
    def __init__(self):
        self.log_reg = None
        self.knn = None
        self.svm = None
        self.nb = None
        self.tree = None
        self.forest = None

    def train_log(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.log_reg = LogisticRegression()
        self.log_reg.fit(X, y)

    def train_knn(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        self.knn.fit(X, y)

    def train_svm(self, X, y):
        from sklearn.svm import SVC
        self.svm = SVC(kernel = 'rbf')
        self.svm.fit(X, y)

    def train_nb(self, X, y):
        from sklearn.naive_bayes import GaussianNB
        self.nb = GaussianNB()
        self.nb.fit(X, y)

    def train_tree(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        self.tree = DecisionTreeClassifier(criterion = 'entropy')
        self.tree.fit(X, y)

    def train_forest(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
        self.forest.fit(X, y)

    def tree_k_fold(self, X, y):
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator=self.tree, X=X, y=y, cv=10)
        return accuracies


def load_pickle(path):
    pkl = open(path, 'rb')
    return pickle.load(pkl)


def save_pickle(obj_path, obj):
    pkl = open(obj_path, "wb")
    pickle.dump(obj, pkl)
    pkl.close()
    return True


def grid_search_fit(X, y, param, mdl):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=mdl,
                               param_grid=param,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)
    return grid_search.fit(X, y)

