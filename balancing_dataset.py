import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where

car_hacking_df = pd.read_csv('/content/car_hacking_data/clean_fuzzy_dataset.csv')

features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']
X = car_hacking_df.loc[:, features].values
X_scaled = StandardScaler().fit_transform(X)

class_le = LabelEncoder()
y = class_le.fit_transform(car_hacking_df['Flag'].values)
counter = Counter(y)
print(f"Before applying NearMiss3: {counter}")

for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

undersample = NearMiss(version=3, n_neighbors_ver3=3, sampling_strategy=1.0)
X_scaled, y = undersample.fit_resample(X_scaled, y)
counter = Counter(y)
print(f"After applying NearMiss3: {counter}")

for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

df = pd.DataFrame (X_scaled, columns = features)
display(df)

df_targets = pd.DataFrame (y, columns=['Flag'])
display(df_targets)

df_col_merged = pd.concat([df, df_targets], axis=1)
display(df_col_merged)


df_col_merged.to_csv('balanced_fuzzy_dataset.csv', index=False)

