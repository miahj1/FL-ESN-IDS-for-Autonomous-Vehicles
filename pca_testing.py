import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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

x = car_hacking_df.loc[:, features].values
y = car_hacking_df.loc[:, ['Flag']].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, 
                            columns=['principal component 1', 'principal component 2'])

final_df = pd.concat([principal_df, car_hacking_df[['Flag']]], axis=1)

# Plotting the graph of the PCA results.

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['R', 'T']
colors = ['g', 'r']

for target, color in zip(targets, colors):
    indices_to_keep = final_df['Flag'] == target
    ax.scatter(final_df.loc[indices_to_keep, 'principal component 1'],
               final_df.loc[indices_to_keep, 'principal component 2'],
               c=color,
               s=50)

ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_

sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar(range(3), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(3))
ax.set_xticklabels(car_hacking_df.columns[1:], rotation=90)
plt.ylim([-1,1])
plt.tight_layout()
plt.show()