# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from metric_tools import tpr_weight_funtion

# In[0]:
with open('data/all_data.pkl', 'rb') as fo:
    all_data_pd = pickle.load(fo)


train_data_pd = all_data_pd[all_data_pd.Tag != -1]
tst_data_pd = all_data_pd[all_data_pd.Tag == -1]

# In[1]:
all_columns = train_data_pd.columns.tolist()
all_columns.remove('UID')
all_columns.remove('Tag')

fea_columns = []

train_data = np.array(train_data_pd[['Tag']].values)
tst_data = np.array(tst_data_pd[['Tag']].values)
for column in all_columns:
    print('train data :', column, train_data_pd[column].values[0])
    fea_columns.extend(['%s_%d' % (column, i) for i in range(len(train_data_pd[column].values[0]))])
    column_train_data = np.array([i for i in train_data_pd[column]])
    train_data = np.concatenate((train_data, column_train_data), axis=1)

    print('tst data :', column, tst_data_pd[column].values[0])
    column_tst_data = np.array([i for i in tst_data_pd[column]])
    tst_data = np.concatenate((tst_data, column_tst_data), axis=1)

print('features num is:', len(fea_columns))

# In[1]:
# y_train_vld = train_data[:, 0]
# x_train_vld = train_data[:, 1:]

x_tst = tst_data[:, 1:]

from sklearn.model_selection import KFold

kf = KFold(n_splits=10,
           # shuffle=True,
           )
kf.get_n_splits(train_data)

final_results = []
vld_scores = []
i = 0
for train_index, vld_index in kf.split(train_data):

    print('start %d data' % i)
    i += 1
    x_train = train_data[train_index][:, 1:]
    y_train = train_data[train_index][:, 0]

    x_vld = train_data[vld_index][:, 1:]
    y_vld = train_data[vld_index][:, 0]

    lgb_train = lgb.Dataset(x_train, y_train,
                            free_raw_data=False)
    lgb_vld = lgb.Dataset(x_vld, y_vld, reference=lgb_train,
                          free_raw_data=False)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'objective': 'rmse',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    early_stopping_rounds=20,
                    valid_sets=lgb_vld,
                    )

    pred_train = gbm.predict(x_train)
    pred_vld = gbm.predict(x_vld)
    pred_tst = gbm.predict(x_tst)

    score_train = tpr_weight_funtion(y_train, pred_train)
    score_vld = tpr_weight_funtion(y_vld, pred_vld)
    print('train score is %.4f, vld score is %.4f' % (score_train, score_vld))
    final_results.append(pred_tst)
    vld_scores.append(score_vld)

# In[2]:

pred_tst = np.mean(final_results, axis=0)
tst_result = tst_data_pd[['UID']]
tst_result['Tag'] = list(pred_tst)
tst_result.to_csv('result/pred_result.csv', index=False)
