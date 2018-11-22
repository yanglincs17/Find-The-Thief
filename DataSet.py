# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[0]:
with open('data/train_data.pkl', 'rb') as fo:
    data_all = pickle.load(fo)

# In[1]:
all_columns = data_all.columns.tolist()
all_columns.remove('UID')
all_columns.remove('Tag')

data = np.array(data_all[['Tag']].values)

for column in all_columns:
    print(column, data_all[column].values[0])
    column_data = np.array([i for i in data_all[column]])
    data = np.concatenate((data, column_data), axis=1)

# In[1]:

y = data[:, 0]
x = data[:, 1:]

x_train, x_vld, y_train, y_vld = train_test_split(x, y, test_size=0.2)








