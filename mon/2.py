import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

import pandas as pd
import numpy as np

import missingno as msno

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

print('얍💢')

data = pd.read_csv('train.csv')
sub = pd.read_csv('test.csv')


y = data['price']
del data['price']


train_len = len(data)
data = pd.concat((data, sub), axis=0)



msno.matrix(data)


for c in data.columns:
    print('{} : {}'.format(c, len(data.loc[pd.isnull(data[c]), c].values)))
    
sub_id = data['id'][train_len:]
del data['id']


print(data.columns)

data['date'] = data['date'].apply(lambda x : str(x[:6]))

print(data.head())

fig, ax = plt.subplots(9, 2, figsize=(12, 50))   # 가로스크롤 때문에 그래프 확인이 불편하다면 figsize의 x값을 조절해 보세요. 

# id 변수(count==0인 경우)는 제외하고 분포를 확인합니다.
count = 1
columns = data.columns
for row in range(9):
    for col in range(2):
        sns.kdeplot(data=data[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count += 1
        if count == 19 :
            break
        
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'sqft_living15']

for c in skew_columns:
    data[c] = np.log1p(data[c].values)

print('얍💢')

fig, ax = plt.subplots(4, 2, figsize=(12, 20))

count = 0
for row in range(4):
    for col in range(2):
        if count == 7:
            break
        sns.kdeplot(data=data[skew_columns[count]], ax=ax[row][col])
        ax[row][col].set_title(skew_columns[count], fontsize=15)
        count+=1
        
sub = data.iloc[train_len:, :]
x = data.iloc[:train_len, :]

print(x.shape)
print(sub.shape)

y = np.log1p(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

lightgbm = LGBMRegressor(random_state=2019)
xgboost = XGBRegressor(random_state=2019)
random_forest = RandomForestRegressor(random_state=2019)

models = [
    {'model': lightgbm, 'name': 'LightGBM'},
    {'model': xgboost, 'name': 'XGBoost'},
    {'model': random_forest, 'name': 'RandomForest'}
]


def get_scores(models, train, y):
    df = {}
    
    for model in models:
        model_name = model.__class__.__name__
        
        X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=random_state, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        df[model_name] = rmse(y_test, y_pred)
        score_df = pd.DataFrame(df, index=['RMSE']).T.sort_values('RMSE', ascending=False)
            
    return score_df

from sklearn.model_selection import GridSearchCV


def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    grid_model = GridSearchCV(model, param_grid=param_grid, \
                            scoring='neg_mean_squared_error', \
                            cv=5, verbose=2, n_jobs=5)
    grid_model.fit(train, y)
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']

    results = pd.DataFrame(params)
    results['score'] = score

    results['RMSE'] = np.sqrt(-1 * results['score']).sort_values
    return(results)

""" def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    grid_model = GridSearchCV(
        model, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error', 
        cv=5, 
        verbose=verbose, 
        n_jobs=n_jobs
    )
    grid_model.fit(train, y)
    
    # 결과 추출
    params = grid_model.cv_results_['params']
    scores = grid_model.cv_results_['mean_test_score']
    
    # DataFrame 생성
    results = pd.DataFrame(params)
    results['score'] = scores
    results['RMSE'] = np.sqrt(-1 * results['score'])  # ✅ .sort_values 제거
    
    results = pd.DataFrame(grid_model.cv_results_['params'])
    results['score'] = grid_model.cv_results_['mean_test_score']
    results['RMSE'] = np.sqrt(-1 * results['score'])
    return results.sort_values(by='RMSE')  """

def save_submission(model, train, y, test, model_name, rmsle=None):
    model.fit(train, y)
    prediction = model.predict(test)

    prediction = np.expm1(prediction)

    mon_dir = os.getenv('USERPROFILE') + '\\Desktop\\aiffel\\mon'

    submission_path = join(mon_dir, 'sample_submission.csv')
    submission = pd.read_csv(mon_dir,'sample_submission.csv')
    submission['price'] = prediction
    submission_csv_path = join(mon_dir, f'submission_{model_name}_RMSLE_{rmsle}.csv')
    submission.to_csv(submission_csv_path, index=False)  # [4]
    print(f'{submission_csv_path} saved!')
    return(save_submission)

save_submission(model, train, y, sub, model_name, '0.164399')