import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# read data
df = pd.read_csv('../../results/sep_dataset.csv')

# plot
df.loc[df['regressor_name'] == 'Baseline', 'feature_generator'] = 'Baseline'
colors = ['C1', 'C0'] * len(df)

plot = df.plot(kind='bar', y=['error', 'baseline'], x='data_source', color=colors, ylim=(0.07, 0.14))

# labels
plt.ylabel('RMSE')
plt.xlabel('Datasource')
plt.savefig('sep_dataset.pdf', format='pdf')
#plt.show()
