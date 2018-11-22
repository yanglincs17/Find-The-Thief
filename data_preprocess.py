# coding: utf-8

import pandas as pd
import numpy as np
import pickle

# In[0]:
transaction_names = [
    'transaction_train_new.csv',
    'transaction_round1_new.csv'
]
operation_names = [
    'operation_train_new.csv',
    'operation_round1_new.csv'
]
tag_names = [
    'tag_train_new.csv'
]

train_trans = pd.read_csv('data/%s' % transaction_names[0])
train_opera = pd.read_csv('data/%s' % operation_names[0])
train_tag = pd.read_csv('data/%s' % tag_names[0])

tst_trans = pd.read_csv('data/%s' % transaction_names[1])
tst_opera = pd.read_csv('data/%s' % operation_names[1])

uid_trans = tst_trans['UID'].unique()
uid_opera = tst_opera['UID'].unique()

tst_uid = np.concatenate((uid_trans, uid_opera))
tst_uid = np.unique(tst_uid)
tst_uid = tst_uid.reshape((tst_uid.shape[0], 1))

# train_trans.sort_values()
opera_columns = train_opera.columns.tolist()
trans_columns = train_trans.columns.tolist()

train_data = train_tag.copy()
tst_data = pd.DataFrame(tst_uid, columns=['UID'])
tst_data['Tag'] = np.ones(tst_uid.shape[0]) * -1

data_all = pd.concat((train_data, tst_data))
trans_all = pd.concat((train_trans, tst_trans))
opera_all = pd.concat((train_opera, tst_opera))

preprocess_data = data_all


# In[1]:
# make feature

opera_time = ['day',
              ]
opera_mode = ['mode', ]
opera_os_ver = ['os', 'version']
opera_device = ['device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1', 'mac2', 'wifi']
opera_wifi = ['ip1', 'ip2', 'ip1_sub', 'ip2_sub']
opera_geo = ['geo_code']
#
opera_unique = opera_os_ver + opera_device + opera_wifi + opera_geo
opera_all_columns = opera_mode + opera_time

trans_unique = [
    'merchant', 'code1', 'code2',
    'acc_id1', 'acc_id2', 'acc_id3',
    'device_code1', 'device_code2', 'device_code3', 'device1', 'device2',
    'mac1', 'ip1', 'geo_code', 'ip1_sub',
    'market_code',
]
trans_all_columns = [
    'channel',
    'trans_type1', 'trans_type2',
    'market_type',
    'amt_src1', 'amt_src2',
    'day', 'trans_amt', 'bal'
]

# In[4]:
all_data_dict = {}
for columns_type in opera_unique + opera_all_columns:
    all_data_dict['opera_%s' % columns_type] = []
for columns_type in trans_unique + trans_all_columns:
    all_data_dict['trans_%s' % columns_type] = []

for i, uid in enumerate(preprocess_data.UID.values[:]):
    uid_opera_data = opera_all[opera_all.UID == uid].sort_values('time').sort_values('day', kind='merge_sort')
    uid_trans_data = trans_all[trans_all.UID == uid].sort_values('time').sort_values('day', kind='merge_sort')

    for columns_type in opera_unique:
        all_data_dict['opera_%s' % columns_type].append(uid_opera_data[columns_type].unique().tolist())
    for columns_type in opera_all_columns:
        all_data_dict['opera_%s' % columns_type].append(uid_opera_data[columns_type].values.tolist())
    for columns_type in trans_unique:
        all_data_dict['trans_%s' % columns_type].append(uid_trans_data[columns_type].unique().tolist())
    for columns_type in trans_all_columns:
        all_data_dict['trans_%s' % columns_type].append(uid_trans_data[columns_type].values.tolist())

    if (i % 1000) == 1:
        print('get %d data feature' % i)


# In[6]:


def fea_one_hot(data_in):
    features = []
    one_hot_list = {}
    for j in data_in:
        for i in j:
            one_hot_list[i] = 0
    one_hot_list = list(one_hot_list.keys())

    print('one hot vector is %d' % len(one_hot_list))
    for i, series in enumerate(data_in):
        fea_vct = [0] * len(one_hot_list)
        for record in series:
            fea_vct[one_hot_list.index(record)] += 1
        features.append(fea_vct)
    return features


def fea_count_times(data_in):
    times_dict = {}
    features = []
    for i, series in enumerate(data_in):
        series = set(series)
        data_in[i] = series
        for record in series:
            if type(record) == str:
                if record in times_dict:
                    times_dict[record] += 1
                else:
                    times_dict[record] = 1
            else:
                if ((record <= 0) | (record >= 0)) == False:
                    continue
                elif record in times_dict:
                    times_dict[record] += 1
                else:
                    times_dict[record] = 1

    for i, series in enumerate(data_in):
        fea = [len(series)]
        ave_times = 0
        max_times = 0
        for record in series:
            if record in times_dict:
                ave_times += times_dict[record]
                if times_dict[record] > max_times:
                    max_times = times_dict[record]
        if fea[0] > 0:
            fea.append(ave_times / fea[0])
            fea.append(max_times)
        else:
            fea.append(0)
            fea.append(0)
        features.append(fea)
    return features


def fea_series_money(data_in):
    features = []
    for i, series in enumerate(data_in):
        if len(series) > 0:
            record = np.array(series)
            max_fea = record.max()
            min_fea = record.min()
            ave_fea = record.mean()
            std_fea = record.std()
            times_fea = record.shape[0]
            fea = [max_fea, min_fea, ave_fea, std_fea, times_fea]
        else:
            fea = [0] * 5
        features.append(fea)
    return features


def fea_time(data_in):
    features = []
    for i, series in enumerate(data_in):
        fea = []
        day_times = {}
        series = np.array(series)
        for record in series:
            if record in day_times:
                day_times[record] += 1
            else:
                day_times[record] = 1

        day_values = list(day_times.values())
        day_keys = list(day_times.keys())
        if len(day_times) > 0:
            fea.append(len(day_times))
            fea.append(max(day_values))
            fea.append(np.mean(day_values))
        else:
            fea.extend([0, 0, 0])
        dis_fea = []
        if len(day_keys) > 1:
            sign_up = [day_keys[i] - day_keys[i - 1] for i in range(1, len(day_keys))]
            dis_fea.append(max(sign_up))
            dis_fea.append(np.mean(sign_up))
            dis_fea.append(np.std(sign_up))
        else:
            dis_fea = [0, 0, 0]
        fea.extend(dis_fea)

        features.append(fea)
    return features


# In[7]:
all_data_features = {}

opera_count_times = opera_device + opera_wifi + opera_geo
opera_one_hot = opera_mode + opera_os_ver

for opera_column in opera_count_times:
    print('get opera %s count times feature' % opera_column)
    all_data_features['opera_%s' % opera_column] = fea_count_times(all_data_dict['opera_%s' % opera_column])
    print('over')

for opera_column in opera_one_hot:
    print('get opera %s one_hot feature' % opera_column)
    all_data_features['opera_%s' % opera_column] = fea_one_hot(all_data_dict['opera_%s' % opera_column])
    print('over')

for opera_column in opera_time:
    print('get opera %s time feature' % opera_column)
    all_data_features['opera_%s' % opera_column] = fea_time(all_data_dict['opera_%s' % opera_column])
    print('over')

# In[8]:

trans_count_times = [
    'merchant', 'code1', 'code2',
    'acc_id1', 'acc_id2', 'acc_id3',
    'device_code1', 'device_code2', 'device_code3', 'device1', 'device2',
    'mac1', 'ip1', 'geo_code', 'ip1_sub',
    'market_code', 'amt_src2', 'market_type', 'trans_type2',
]
trans_one_hot = [
    'channel',
    'trans_type1',
    'amt_src1',
]

trans_series = [
    'trans_amt', 'bal'
]

trans_time = [
    'day'
]

for trans_column in trans_count_times:
    print('get opera %s count times feature' % trans_column)
    all_data_features['trans_%s' % trans_column] = fea_count_times(all_data_dict['trans_%s' % trans_column])
    print('over')

for trans_column in trans_one_hot:
    print('get opera %s one_hot feature' % trans_column)
    all_data_features['trans_%s' % trans_column] = fea_one_hot(all_data_dict['trans_%s' % trans_column])
    print('over')

for trans_column in trans_series:
    print('get opera %s series feature' % trans_column)
    all_data_features['trans_%s' % trans_column] = fea_series_money(all_data_dict['trans_%s' % trans_column])
    print('over')

for trans_column in trans_time:
    print('get opera %s times feature' % trans_column)
    all_data_features['trans_%s' % trans_column] = fea_time(all_data_dict['trans_%s' % trans_column])
    print('over')

# In[9]:

all_columns = list(all_data_features.keys())

for select_column in all_columns:
    print(select_column)
    preprocess_data[select_column] = all_data_features[select_column]
    print('over')

# In[10]:

with open('data/all_data.pkl', 'wb') as fw:
    pickle.dump(preprocess_data, fw)



