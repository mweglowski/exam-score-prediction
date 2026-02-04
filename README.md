# Student Exam Score Prediction
Project is about prediction of exam scores of students. Idea derives from latest [kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e1/overview). In addition frontend in `React` has been created as well as REST API in `Flask` for simple models usage.

## Preview
![](images/preview.jpg)

## Configuration
Project has been made using `Python` 3.12.10 and `Node.js` v22.15.0. Ensure that you also have `npm` installed.

Ensure to install `requirements.txt` by running the following command.
```bash
pip install -r requirements.txt
```

Download `.csv` data from [kaggle](https://www.kaggle.com/competitions/playground-series-s6e1/data). Ensure that it is placed inside `/data` folder.


## Libraries
Below are presented python packages used for that project:
* `numpy` - handling numerical data
* `pandas` - working with tables
* `sklearn` - machine learning tools
* `lightgbm` - gradient boosting library
* `optuna` - model tuning

## Project Structure
```txt

cv_models
images
main.py
preprocessing.py
tools.py
eda.ipynb
requirements.txt
experiments.md
```

| Component | Description |
| data | Directory in which `.csv` files are stored |
| cv_models | Stores saved models from experiments |
| images | Keeps graphs from data analysis and tuning history |
| main.py | Core file to perform experiment on specific model |
| preprocessing.py | 



## Data Analysis
Firstly I have made some exploratory data analysis to get more familiar with dataset. Data consists of 13 columns, either numerical or categorical. In `eda.ipynb` there are lots of plots showing data distribution and correlations.

Here are a few plots showing relation of some features to the target `exam_score`:
![]()

![]()

![]()

## Preprocessing
Numerical columns have been scaled using sklearn's `StandardScaler` to perform `z = (x - u) / s` on each sample. Categorical columns were divided into ordinal as well as not related to each other, there were used respectively sklearn's `OrdinalEncoder` and `OneHotEncoder`.

## Models
There were lots of models used for experiments. Starting from simple linear regression as a **baseline** which turned out to be really good ending on **gradient boosting methods** like `LightGBM`.

Models tested (chronologically):
* `LinearRegression`
* `DecisionTreeRegressor`
* `RandomForest`
* `AdaBoostRegressor`
* `GradientBoostingRegressor`
* `HistGradientBoostingRegressor`
* `LightGBMRegressor`

`LightGBMRegressor` turned out to perform the best, so that I decided to put more time on it and leverage `optuna` to find the best parameters and improve it even more.

## Tuning
I wanted to use grid search for tuning parameters of `RandomForest`, but resigned due to RAM problems. Due to the fact that dataset is so big - hunders of thousands samples, this algorithm consumed so much memory.

Fortunately gradient boosting techniques turned out to be better. They consumed less memory, time and achieved better RMSE.

I have been also tuning `LightGBM` regressor for 50 trials. Each consisting of `KFold` cross-validation splitted into 5 folds. This was performed by usage of `optuna` and goal was to minimize RMSE. Best achieved values:

**Score**: `~8.7881`
```python
{
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
```

**Optimization history**
![](images/optimization_history.jpg)