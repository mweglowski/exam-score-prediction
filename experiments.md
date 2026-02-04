# Experiment 1
> First experiment with linear regression - our **baseline**.

**Mean score**: 8.8953

**Time elapsed**: 11.18s

```python
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num', StandardScaler(),
                                                  ['age', 'study_hours',
                                                   'class_attendance',
                                                   'sleep_hours']),
                                                 ('cat_onehot',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse_output=False),
                                                  ['study_method', 'course',
                                                   'gender']),
                                                 ('cat_ordinal',
                                                  OrdinalEncoder(categories=[['low',
                                                                              'medium',
                                                                              'high'],
                                                                             ['poor',
                                                                              'average',
                                                                              'good'],
                                                                             ['easy',
                                                                              'moderate',
                                                                              'hard'],
                                                                             ['no',
                                                                              'yes']]),
                                                  ['facility_rating',
                                                   'sleep_quality',
                                                   'exam_difficulty',
                                                   'internet_access'])])),
                ('regressor', LinearRegression())])
```

# Experiment 2
> Testing decision tree

**Mean score**: 12.8269

**Time elapsed**: 51.97s

```python
Pipeline(steps=[('preprocessor', ...)
                ('regressor', DecisionTreeRegressor())])
```

# Experiment 3
> Testing random forest with reduced estimators number and 3 folds in cross-validation

**Mean score**: 9.2022

**Time elapsed**: 92.38s

```python
Pipeline(steps=[('preprocessor', ...)
                ('regressor',
                 RandomForestRegressor(n_estimators=30, n_jobs=4))])
```

# Experiment 4
> Increased n_jobs 4 -> 6 and n_estimators 30->50, still 3 folds

**Mean score**: 9.1439

**Time elapsed**: 110.14s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 RandomForestRegressor(n_estimators=50, n_jobs=6))])
```

# Experiment 5
> n_jobs 6 -> 8, max_depth restricted to 10

**Mean score**: 9.1255

**Time elapsed**: 49.26s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 RandomForestRegressor(max_depth=10, n_estimators=50,
                                       n_jobs=8))])
```

# Experiment 6
> n_estimators 50 -> 100

**Mean score**: 9.1235

**Time elapsed**: 80.34s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', RandomForestRegressor(max_depth=10, n_jobs=8))])
```

# Experiment 7
> n_estimators 100 -> 200

**Mean score**: 9.1237

**Time elapsed**: 152.75s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 RandomForestRegressor(max_depth=10, n_estimators=200,
                                       n_jobs=8))])
```

# Experiment 8
> max_depth 10 -> 20

**Mean score**: 9.0341

**Time elapsed**: 269.93s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 RandomForestRegressor(max_depth=20, n_estimators=200,
                                       n_jobs=8))])
```

# Experiment 9
> max_depth 20 -> None

**Mean score**: 9.0721

**Time elapsed**: 269.52s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 RandomForestRegressor(n_estimators=200, n_jobs=10))])
```

# Experiment 10
> Starting experimenting with boosting algorithms - AdaBoost with 20 trees

**Mean score**: 10.1889

**Time elapsed**: 87.01s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', AdaBoostRegressor(n_estimators=20))])
```

# Experiment 11
> trees 20 -> 50

**Mean score**: 9.9963

**Time elapsed**: 209.53s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', AdaBoostRegressor())])
```

# Experiment 12
> trees 50 -> 100

**Mean score**: 10.0023

**Time elapsed**: 297.22s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', AdaBoostRegressor(n_estimators=100))])
```

# Experiment 13
> Trying sklearn's plain gradient boosting regressor

**Mean score**: 8.8554

**Time elapsed**: 332.28s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', GradientBoostingRegressor())])
```

# Experiment 14
> Check hist gradient boosting

**Mean score**: 8.8170

**Time elapsed**: 17.82s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor', HistGradientBoostingRegressor())])
```
# Experiment 15
> Starting experimenting with LightGBM. Currently the best score. Time duration is almost the same as for simple linear regression from first experiment.

**Mean score**: 8.8135

**Time elapsed**: 11.85s

```python
Pipeline(steps=[('preprocessor', preprocessor),
                ('regressor', LGBMRegressor())])
```

# Experiment 16
> Testing tuned LightGBM

**Mean score**: 8.7644

**Time elapsed**: 24.41s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 LGBMRegressor(bagging_fraction=0.8986253939886042,
                               bagging_freq=5,
                               colsample_bynode=0.8685671163349028,
                               colsample_bytree=0.670750939608245,
                               lambda_l1=6.336661658433877,
                               lambda_l2=5.713800450906738,
                               learning_rate=0.09854420698068374, max_depth=8,
                               min_data_in_leaf=77, num_leaves=250,
                               num_trees=200, objective='regression',
                               verbosity=-1))])
```

# Experiment 17
> Testing 3 tuned LightGBM using VotingRegressor

**Mean score**: 8.7644

**Time elapsed**: 65.02s

```python
VotingRegressor(estimators=[('lightgbm1',
                             Pipeline(steps=[('preprocessor', ...),
                                             ('regressor',
                                              LGBMRegressor(bagging_fraction=0.8986253939886042,
                                                            bagging_freq=5,
                                                            colsample_bynode=0.8685671163349028,
                                                            colsample_bytree=0.670750939608245,
                                                            lambda_l1=6.336661658433877,
                                                            lambda_l2=5.713800450906738,
                                                            learning_rate=0.09854420698068374,
                                                            max_depth=8,
                                                            min_data_in_leaf=77,
                                                            num_leaves=250,
                                                            num_trees=200,
                                                            objective='regression',
                                                            verbosity=-1))]))])
```

# Experiment 18
> Another time

**Mean score**: 8.7644

**Time elapsed**: 23.46s

```python
Pipeline(steps=[('preprocessor', ...),
                ('regressor',
                 LGBMRegressor(bagging_fraction=0.8986253939886042,
                               bagging_freq=5,
                               colsample_bynode=0.8685671163349028,
                               colsample_bytree=0.670750939608245,
                               lambda_l1=6.336661658433877,
                               lambda_l2=5.713800450906738,
                               learning_rate=0.09854420698068374, max_depth=8,
                               min_data_in_leaf=77, num_leaves=250,
                               num_trees=200, objective='regression',
                               verbosity=-1))])
```

