import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

car_hacking_df = pd.read_csv('/content/car_hacking_data/DoS_dataset.csv',
                                              names=['Timestamp', 'CAN_ID', 'DLC', 'D0', 
                                                     'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                                                     'D7', 'Flag'])

def hex_to_float(value):
    return float.fromhex(value)

car_hacking_df['CAN_ID'] = car_hacking_df['CAN_ID'].apply(hex_to_float)

car_hacking_df['Data'] = (car_hacking_df['D0'].astype(str).replace("nan","").replace("R", "") + 
                          car_hacking_df['D1'].astype(str).replace("nan","").replace("R", "") + 
                          car_hacking_df['D2'].astype(str).replace("nan","").replace("R", "") + 
                          car_hacking_df['D3'].astype(str).replace("nan","").replace("R", "") + 
                          car_hacking_df['D4'].astype(str).replace("nan","").replace("R", "") +
                          car_hacking_df['D5'].astype(str).replace("nan","").replace("R", "") + 
                          car_hacking_df['D6'].astype(str).replace("nan","").replace("R", "") +
                          car_hacking_df['D7'].astype(str).replace("nan","").replace("R", ""))

car_hacking_df = car_hacking_df.drop(columns=['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'])

car_hacking_df['Data'] = car_hacking_df['Data'].apply(hex_to_float)

car_hacking_df = car_hacking_df.fillna(car_hacking_df['Flag'].mode().iloc[0])

features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']

X_train = car_hacking_df.loc[:, features].values
y_train = car_hacking_df.loc[:, ['Flag']].values

X_train_reduced = StandardScaler().fit_transform(X_train)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(X_train)
rnd_frcl = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time()
rnd_frcl.fit(X_train_reduced, y_train)
t1 = time.time()

print(f'Training took {t1 - t0}s')

importances = rnd_frcl.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print(f'{f+1}) ')