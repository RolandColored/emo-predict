import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# read data
df = pd.read_csv('../../results/bow.csv')

# filter and sort
df = df[df['data_source'].str.contains('foxnews')]
df = df.sort_values('error')

# plot
df.loc[df['regressor_name'] == 'Baseline', 'feature_generator'] = 'Baseline'
df['feature_generator'] = df['feature_generator'].str.replace('text', '').str.replace('_', ' ').str.title()
colors = ['C1'] * (len(df) - 1) + ['C0']
df.plot(kind='bar', y='error', x='feature_generator', legend=False, color=colors, ylim=(0.09, 0.12))

# labels
plt.ylabel('RMSE')
plt.xlabel('Feature Ausprägung')
plt.savefig('bow_en.pdf', format='pdf')
#plt.show()
