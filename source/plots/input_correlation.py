import glob

import pandas as pd

# gather data
df_all = pd.DataFrame()
for file in glob.glob("../../fb-data/*.csv"):
    df = pd.read_csv(file, index_col='id')
    df = df.drop(['link', 'message', 'date'], axis=1)
    df_all = df_all.append(df)

print(df_all.corr().to_latex())
