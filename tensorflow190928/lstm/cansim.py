import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('./data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9, engine='python')
# print(df.head())
df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
# print(df.head())
# plt.plot(df)
# plt.show()
split_date = pd.Timestamp('01-01-2011')
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[:split_date, ['Unadjusted']]
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train','test'])
# plt.plot()
# plt.show()
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
# print(train_sc_df.head())
# pandas shift를 통해 window 만들기
# shift는 이전 정보를 다음 row에서 다시 쓰기 위한 pandas 함수
# 이 작업의 이유는 과거값 shift1~12를 통해 현재값 Scaled를 예측하는 것
for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['test_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
# print(train_sc_df.head(13))
x_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]
x_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]
#최종 트레이닝 셋
print(x_train.head())
print(y_train.head())
# ndarray로 변환
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
print(x_train.shape)
print(x_train)
print(y_train.shape)
print(y_train)
print('**************************')
x_train_t = x_train.reshape(x_train.shape[0], 12, 1)
x_test_t = x_train.reshape(x_test.shape[0], 12, 1)
print('최종 Data')
print(x_train.shape)
print(x_train)
print(y_train.shape)
# LSTM 모델 만들기
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential()
model.add(LSTM(20, input_shape=(12, 1))) #(timestamp, feature)
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer='adam')

# print(model.summary())

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
model.fit(x_train_t, y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

print('******** x_test-t **********')
print(x_test_t)
print('******** 모델 예측값 **********')
y_pred = model.predict(x_test_t)
print(y_pred)
