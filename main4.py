from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest

import otra_forma.io_utiil as io

X , submit  = io.load_dataframes('train.csv','test.csv')

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

y = X[['ID','race_group','efs','efs_time']]


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

    rsf = RandomSurvivalForest(n_estimators=2, n_jobs=4,low_memory=True)
    io.train(rsf,X_fold_train.drop(['ID','efs','efs_time','race_group'],axis=1))
    predict = rsf.predict(X_fold_val.drop(['ID','efs','efs_time','race_group'],axis=1))
    score = io.get_score(y_fold_val,predict)

    print('scoring = '+score)