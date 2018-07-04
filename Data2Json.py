from sklearn.datasets import load_wine
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import sys

data = load_wine()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['cid'] = data['target'].astype(float)

features = list(df)
x = df.loc[:, features].values
y = df.loc[:,['cid']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['x', 'y', 'z'])

finalDf = pd.concat([principalDf, df[['cid']]], axis = 1)

output_file = 'data.json'

d_json = { 'points': finalDf.to_json( None, orient= 'records' )}
with open(output_file, 'w') as outfile:
    json.dump(d_json, outfile)
