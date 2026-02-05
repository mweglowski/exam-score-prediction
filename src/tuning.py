from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np
import lightgbm
import optuna

from src.tools import load_data, get_features_and_labels
from src.preprocessing import get_preprocessor


def perform_grid_search_on_random_forest(estimator, X, y):
    param_grid = { # added regressor__ because estimator is a Pipeline
        'regressor__n_estimators': [50, 150],
        'regressor__max_depth': [None, 10, 20],
        'regressor__max_features': [1.0, 'sqrt'], # number of features to consider when looking for the best split if log2: max_features=log2(n_features)
        'regressor__min_samples_leaf': [1, 4] # required number of samples to be at a leaf node
    }

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=6, verbose=3, scoring='neg_root_mean_squared_error')

    grid_search.fit(X, y)
    print(f'Best Params: {grid_search.best_params_}')
    print(f'Best RMSE: {-grid_search.best_score_}')

def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'verbosity': -1, # whether info is shown
        'num_trees': 200, # number of trees in the model
        'boosting_type': 'gbdt', # type of boosting (gbdt, rf, dart)
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True), # regularization (Lasso), adds penalty for non-zero weights, encourage to use fewer features
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True), # (Ridge) adds penalty for large weights, encourages small, stable weights, prevents from domination of some features
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True), # how much should new tree affect previous results
        'max_depth': trial.suggest_int('max_depth', 4, 8), # max depth of a tree
        'max_leaves': trial.suggest_int('num_leaves', 16, 256), # num of leaves in one tree, it should be < 2**max_depth
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0), # for each tree select only x% of features
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0), # for each split (node) in a tree select x% of available features
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0), # for each iteration use only x% trainig data rows
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), # per how many trees data should be reshuffled
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 5, 100), # min num of samples required for leaf creation in a single tree
        'device_type': 'gpu',
    }

    preprocessor = get_preprocessor()
    estimator = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lightgbm.LGBMRegressor(**params)),
    ])

    df_train, _ = load_data()
    X, y = get_features_and_labels(df=df_train,
                                   target_col='exam_score')

    kf = KFold(n_splits=5, shuffle=True, random_state=303)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print('Fold:', fold)
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        estimator.fit(X_train_fold, y_train_fold)
        preds = estimator.predict(X_val_fold)
        score = root_mean_squared_error(y_val_fold, preds)
        scores.append(score)

    print(f'Scores: {scores}')
    return np.mean(scores)

def perform_study():
    study = optuna.create_study(direction='minimize')
    study.optimize(lgb_objective, n_trials=50)
    print('Best score:', study.best_value)
    print('Best params:', study.best_params)

def main():
    perform_study()

if __name__ == '__main__':
    main()