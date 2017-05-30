import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# read data
df = pd.read_csv('../../results/features.csv')

# filter and sort
df = df[df['data_source'].str.contains('foxnews')]

# prepare data
df = df[['error', 'feature_generator', 'regressor_name']]
df = df.groupby('regressor_name').mean()
df = df.sort_values('error')

print(df)

# set other color for baseline
colors = ['C1'] * len(df)
for i, row in enumerate(df.index):
    if row == 'Baseline':
        colors[i] = 'C0'

# plot
plot = df.plot.bar(legend=False, color=colors, ylim=(0.11, 0.18))

# labels
plt.ylabel('Mean RMSE per Feature')
plt.xlabel('Regressor')
#plt.savefig('features_mean_en.pdf', format='pdf')
