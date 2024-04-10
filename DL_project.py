from sqlalchemy import create_engine
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import talib
from tqdm import tqdm

# DB 연결
engine = create_engine("mysql+pymysql://root:6826@127.0.0.1:3306/stock_db")
con = pymysql.connect(user = "root",
                      passwd = "6826",
                      host = "127.0.0.1",
                      db = "stock_db",
                      charset = "utf8")
mycursor = con.cursor()

# data = pd.read_sql("""select * from kor_price
#                     where 종목코드 = (select 종목코드 from kor_ticker
#                     where 종목명 = "삼성전자");""", con = engine)

data = pd.read_sql("""SELECT * FROM kor_price
                   where 날짜 > "2018-01-01" and
                   종목코드 IN (SELECT 종목코드
                   FROM (SELECT 종목코드
                         FROM kor_ticker
                         ORDER BY 시가총액 DESC
                         LIMIT 100) AS t);""", con = engine)

engine.dispose()
con.close()

data_code = set(data["종목코드"].values)
final_df = pd.DataFrame()
for code in tqdm(data_code):
    temp_data = pd.DataFrame(data[data["종목코드"] == code]["종가"]).reset_index()

    # 단순 이동평균
    temp_data["SMA_5"] = talib.SMA(np.array(data[data["종목코드"] == code]["종가"]), 5)
    temp_data["SMA_20"] = talib.SMA(np.array(data[data["종목코드"] == code]["종가"]), 20)
    temp_data["SMA_60"] = talib.SMA(np.array(data[data["종목코드"] == code]["종가"]), 60)
    temp_data["SMA_120"] = talib.SMA(np.array(data[data["종목코드"] == code]["종가"]), 120)

    # 지수 이동평균
    temp_data["EMA_5"] = talib.EMA(np.array(data[data["종목코드"] == code]["종가"]), 5)
    temp_data["EMA_20"] = talib.EMA(np.array(data[data["종목코드"] == code]["종가"]), 20)
    temp_data["EMA_60"] = talib.EMA(np.array(data[data["종목코드"] == code]["종가"]), 60)
    temp_data["EMA_120"] = talib.EMA(np.array(data[data["종목코드"] == code]["종가"]), 120)

    # 상대 강도 지수 (RSI)
    temp_data["RSI_14"] = talib.RSI(np.array(data[data["종목코드"] == code]["종가"]), 14)

    # 볼린저밴드
    upper_2sd, mid_2sd, lower_2sd = talib.BBANDS(np.array(data[data["종목코드"] == code]["종가"]), 
                                                nbdevup = 2,
                                                nbdevdn = 2,
                                                timeperiod = 20)

    bb = pd.concat([pd.Series(upper_2sd), pd.Series(mid_2sd), pd.Series(lower_2sd)], axis = 1)
    bb.columns = ["Upper Band", "Mid Band", "Lower Band"]

    df = pd.concat([temp_data.iloc[120:, :], bb.iloc[120:, :]], axis = 1)

    final_df = pd.concat([final_df, df], axis = 0)
 
df = final_df

df['Target'] = ((df['종가'].shift(-1) / df['종가']) - 1) > 0.03
df['Target'] = df['Target'].astype(int)

# Initialize an empty list to hold our reshaped data
reshaped_features = []
reshaped_targets = []

# Loop through the DataFrame, starting from the 10th day to the second to last day (since we're predicting D+1)
for i in tqdm(range(10, len(df) - 1)):
    # Extract the last 10 days + current day for 'Close' and each EMA
    window_close = df['종가'][i-10:i+1].to_list()
    window_sma_5 = df['SMA_5'][i-10:i+1].to_list()
    window_sma_20 = df['SMA_20'][i-10:i+1].to_list()
    window_sma_60 = df['SMA_60'][i-10:i+1].to_list()
    window_sma_120 = df['SMA_120'][i-10:i+1].to_list()

    window_ema_5 = df['EMA_5'][i-10:i+1].to_list()
    window_ema_20 = df['EMA_20'][i-10:i+1].to_list()
    window_ema_60 = df['EMA_60'][i-10:i+1].to_list()
    window_ema_120 = df['EMA_120'][i-10:i+1].to_list()

    window_RSI = df["RSI_14"][i-10:i+1].to_list()
    window_upper = df["Upper Band"][i-10:i+1].to_list()
    winow_mid = df["Mid Band"][i-10:i+1].to_list()
    window_lower = df["Lower Band"][i-10:i+1].to_list()

    # Flatten this window of data and append to our reshaped features list
    feature_row = (window_close + window_sma_5 + window_sma_20 + window_sma_60 + window_sma_120 +
                window_ema_5 + window_ema_20 + window_ema_60 + window_ema_120 + window_RSI + 
                window_upper + winow_mid + window_lower)
    reshaped_features.append(feature_row)
    
    # Append the target variable for this day
    reshaped_targets.append(df['Target'].iloc[i])

# Convert the lists into a DataFrame
reshaped_df = pd.DataFrame(reshaped_features)
reshaped_df['Target'] = reshaped_targets

reshaped_df["Target"].value_counts()

# 컬럼 이름 생성기
col_names = []
indic_dict = {1:"close", 2:"sma_5", 3:"sma_20", 4:"sma_60", 5:"sma_120", 
              6:"ema_5", 7:"ema_20", 8:"ema_60", 9:"ema_120", 10:"rsi_14", 
              11:"upper_band", 12:"mid_band", 13:"lower_band"}
cnt = 1
for i in range(13):
    for j in range(10, 0, -1):
        tmp = f"D-{j}_{indic_dict[cnt]}"
        col_names.append(tmp)
    col_names.append(f"D-day_{indic_dict[cnt]}")
    cnt += 1 

len(col_names)

X = reshaped_df.iloc[:, :-1]
y = reshaped_df.iloc[:, -1]

X.columns = col_names

# 10-window min-max normalizarion
def min_max_10_window(row):
    max_v = max(row)
    min_v = min(row)
    if max_v == min_v:
        return 0
    return ((row - min_v) / (max_v - min_v))

X_temp = pd.DataFrame()
for i in tqdm(range(11, 144, 11)):
    tmp = X.iloc[:, i - 11:i].apply(min_max_10_window, axis = 1)
    X_temp = pd.concat([X_temp, tmp], axis = 1)
    
X = X_temp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scaler = StandardScaler()
# scaler.fit_transform(X_train)
# scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

model = MLPClassifier(hidden_layer_sizes = (16, 16, 16, 16), activation = "relu", 
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

with open('C:/Users/pamda/OneDrive/바탕 화면/python/output1.txt', 'a', encoding='utf-8') as file:
    file.write(f"\n24_04_10_ver2\n{cm1_rp}")

with open('C:/Users/pamda/OneDrive/바탕 화면/python/output2.txt', 'a', encoding='utf-8') as file:
    file.write(f"\n24_04_10_ver2\n{cm2_rp}" )