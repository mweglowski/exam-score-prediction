from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
import lightgbm
import joblib
import time
import os

from src.tools import load_data, get_features_and_labels
from src.preprocessing import get_preprocessor


def get_cv_results(estimator, estimator_name, X, y, exp_num):
    start_time = time.perf_counter()
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=303)

    oof_preds = np.zeros(X.shape[0])
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print('Fold:', fold)
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        estimator.fit(X_train_fold, y_train_fold)
        preds = estimator.predict(X_val_fold)
        oof_preds[val_idx] = preds
        score = root_mean_squared_error(y_val_fold, preds)
        scores.append(score)

        if fold == 0:
            joblib.dump(estimator, f'models/{estimator_name}_{exp_num}.pkl')

    end_time = time.perf_counter()
    time_elapsed_in_seconds = round(end_time - start_time, 2)

    print(f'Scores: {scores}')
    print(f'Duration: {time_elapsed_in_seconds}s')
    return np.mean(scores), time_elapsed_in_seconds

def save_experiment(estimator, mean_cv_score, exp_num, changes, filename, time_elapsed):
    if os.path.exists(filename):
        with open(filename, 'a') as file:
            content = f'# Experiment {exp_num}\n'
            content += f'> {changes}\n\n'
            content += f'**Mean score**: {mean_cv_score:.4f}\n\n'
            content += f'**Time elapsed**: {time_elapsed}s\n\n'
            # content += f'**Test RMSE**: {test_score:.4f}\n'
            content += f'```python\n{estimator}\n```\n\n'
            file.write(content)
    else:
        raise FileNotFoundError(f'File {filename} does not exist')

def main():
    tuned_lightgbm_params = {
        'objective': 'regression',
        'verbosity': -1,
        'num_trees': 200,
        'boosting_type': 'gbdt',
        'lambda_l1': 6.336661658433877, 
        'lambda_l2': 5.713800450906738, 
        'learning_rate': 0.09854420698068374, 
        'max_depth': 8, 
        'num_leaves': 250, 
        'colsample_bytree': 0.670750939608245,
        'colsample_bynode': 0.8685671163349028, 
        'bagging_fraction': 0.8986253939886042, 
        'bagging_freq': 5, 
        'min_data_in_leaf': 77
    }

    preprocessor = get_preprocessor()
    estimators = {
        'linear_regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'decision_tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor())
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                n_jobs=10,
            ))
        ]),
        'ada_boost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', AdaBoostRegressor(
                n_estimators=100,
            ))
        ]),
        'gradient_boosting_sklearn': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor())
        ]),
        'hist_gradient_boosting_sklearn': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor()) # faster, better for large datasets >= 10 000 samples, inspired by LightGBM, built-in support for missing values and handling categorical data
        ]),
        'lightgbm': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lightgbm.LGBMRegressor())
        ]),
        'tuned_lightgbm': VotingRegressor([
            ('lightgbm1', Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', lightgbm.LGBMRegressor(**tuned_lightgbm_params))
            ])),
            ('lightgbm2', Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', lightgbm.LGBMRegressor(**tuned_lightgbm_params))
            ])),
            ('lightgbm3', Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', lightgbm.LGBMRegressor(**tuned_lightgbm_params))
            ])),
        ]),
        'tuned_lightgbm': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lightgbm.LGBMRegressor(**tuned_lightgbm_params))
        ]),
    }
    
    EXP_NUM = 19
    df_train, _ = load_data()
    X, y = get_features_and_labels(df=df_train,
                                   target_col='exam_score')

    estimator_name = 'tuned_lightgbm'
    estimator = estimators[estimator_name]

    score, time_elapsed = get_cv_results(estimator, estimator_name, X, y, EXP_NUM)

    changes = 'Another time'

    save_experiment(estimator=estimator,
                    mean_cv_score=score,
                    exp_num=EXP_NUM,
                    changes=changes,
                    filename='experiments.md',
                    time_elapsed=time_elapsed)

if __name__ == '__main__':
    main()