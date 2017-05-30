import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# read data
df = pd.read_csv('../../results/features_combined.csv')

# filter and sort
df = df[df['data_source'].str.contains('bild')]
df = df.sort_values('error')

# plot
df.loc[df['regressor_name'] == 'Baseline', 'feature_generator'] = 'Baseline'
df['feature_generator'] = df['feature_generator']\
    .str.replace('_', ' ')\
    .str.replace(' and', ',')\
    .str.replace('depechemood', 'depmd.')\
    .str.replace('e, t', 'e\nt')\
    .str.title()
colors = ['C1'] * len(df)

# set other color for baseline
for i, row in enumerate(df['regressor_name']):
    if row == 'Baseline':
        colors[i] = 'C0'

plot = df.plot(kind='bar', y='error', x='feature_generator', legend=False, color=colors, ylim=(0.09, 0.14))

# labels
plt.ylabel('RMSE')
plt.xlabel('Feature')
plt.savefig('features_combined_de.pdf', format='pdf')
#plt.show()
