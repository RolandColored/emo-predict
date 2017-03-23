import glob
import ntpath

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# gather data
df_summed = pd.DataFrame()
for file in glob.glob("../../fb-data/*.csv"):
    df = pd.read_csv(file, index_col='id')
    df = df.drop(['link', 'message', 'date'], axis=1)

    df_row = df.sum() / df.sum().sum()
    df_row['dataset'] = ntpath.basename(file).replace('.csv', '')
    df_summed = df_summed.append(df_row, ignore_index=True)

df_summed = df_summed.set_index('dataset').transpose()
df_summed.plot(kind='bar')

# labels
plt.ylabel('Anteil')
plt.xlabel('Label')
frame = plt.legend().get_frame()
frame.set_facecolor('white')
plt.savefig('input_distribution.pdf', format='pdf', bbox_inches='tight')
