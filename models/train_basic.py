import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb

datadir = '/mnt/liberty-mutual-group-property-inspection-prediction'
train = pd.read_csv(f'{datadir}/train.csv', index_col=0)
test = pd.read_csv(f'{datadir}/test.csv', index_col=0)

y = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

columns = train.columns

X_train, X_test = np.array(train), np.array(test)

# some variables are numeric, some variables are string categorical, we need to label encode the
# categorical ones
for i in range(X_train.shape[1]):
    if type(X_train[1, i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[:, i]) + list(X_test[:, i]))
        X_train[:, i] = lbl.transform(X_train[:, i])
        X_test[:, i] = lbl.transform(X_test[:, i])

X_train, X_test = X_train.astype(float), X_test.astype(float)

params = {
    "objective": "reg:squarederror",
    "eta": 0.01,
    "min_child_weight": 5,
    "subsample": 0.8,
    "scale_pos_weight": 1.0,
    "max_depth": 7,
    "verbosity": 1
}
params = list(params.items())

# Use 5000 rows (~10% of the dataset) as the validation set.
X_valid_split_idx = 5000
num_rounds = 2000

# create train, validation, and test dmatrices
xgtrain = xgb.DMatrix(X_train[X_valid_split_idx:,:], label=y[X_valid_split_idx:])
xgval = xgb.DMatrix(X_train[:X_valid_split_idx,:], label=y[:X_valid_split_idx])
xgtest = xgb.DMatrix(X_test)

# train using early stopping and predict
watchlist = [(xgtrain, 'train'), (xgval, 'val')]
model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)

# Generate predictions and save them to disk.
y_test_pred = model.predict(xgtest)
(pd.DataFrame({"Id": test.index, "Hazard": y_test_pred})
 .set_index("Id")
 .to_csv("/spell/predictions.csv"))
