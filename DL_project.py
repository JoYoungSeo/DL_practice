from sqlalchemy import create_engine
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf; tf.keras
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras import Sequential
# from tensorflow.keras.activations import sigmoid
# from sklearn.metrics import mean_squared_error

# DB 연결
engine = create_engine("mysql+pymysql://root:6826@127.0.0.1:3306/stock_db")
con = pymysql.connect(user = "root",
                      passwd = "6826",
                      host = "127.0.0.1",
                      db = "stock_db",
                      charset = "utf8")
mycursor = con.cursor()

data = pd.read_sql("""select * from kor_price
                    where 종목코드 = (select 종목코드 from kor_ticker
                    where 종목명 = "삼성전자");""", con = engine)
engine.dispose()
con.close()

ma = [5,20,60,120]

for days in ma:
    data['ma_' + str(days)] = data['종가'].rolling(window = days).mean()

data["over_1bil"] = data.apply(lambda x: 1 if x["종가"] * x["거래량"] > 10 ** 9 else 0, axis = 1)

data["y"] = 0
for i in range(1, 11):
    data[f"D-{i}"] = 0
    data[f"D-{i}_ma_5"] = 0
    data[f"D-{i}_ma_20"] = 0
    data[f"D-{i}_ma_60"] = 0
    data[f"D-{i}_ma_120"] = 0

for i in range(130, len(data) - 1):
    if data["over_1bil"][i]:
        
        # D+1일의 종가가 3% 이상 상승했으면
        if data["종가"][i + 1] >= data["종가"][i] * 1.03:
            data["y"][i] = 1

    # D-day를 기준으로 10일치의 주가 데이터 추가
    for j in range(1, 11):
        data[f"D-{j}"][i] = data["종가"][i - j]
        data[f"D-{j}_ma_5"][i] = data["ma_5"][i - j]
        data[f"D-{j}_ma_20"][i] = data["ma_20"][i - j]
        data[f"D-{j}_ma_60"][i] = data["ma_60"][i - j]
        data[f"D-{j}_ma_120"][i] = data["ma_120"][i - j]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

using_cols = ['종가', '거래량', 'ma_5', 'ma_20', 'ma_60','ma_120']

X = data[using_cols].iloc[130:-1]
y = data["y"].iloc[130:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

model = MLPClassifier(hidden_layer_sizes = (7, 7, 7), activation = "relu", 
                      solver = "adam", alpha = 0.0001, batch_size = 28, 
                      learning_rate = "constant", learning_rate_init = 0.001, 
                      power_t = 0.5, max_iter = 2000, random_state = 42)

model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)
print("오버샘플링 적용 X\n", cm1)
cm1_rp = classification_report(y_test, y_pred1)
print(cm1_rp)

model.fit(X_train_over, y_train_over)
y_pred2 = model.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
print("오버샘플링 적용 O\n", cm2)
cm2_rp = classification_report(y_test, y_pred2)
print(cm2_rp)