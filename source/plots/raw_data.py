import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# plot
df_ridge = pd.read_csv('../../results/raw_predictions/raw_data_en.csv')
df_ridge['source'] = 'Ridge'
df_baseline = pd.read_csv('../../results/raw_predictions/raw_data_baseline_en.csv')
df_baseline['source'] = 'Baseline'
df_all = df_ridge.append(df_baseline)

plots = df_all.groupby('source').boxplot(showmeans=True, flierprops=dict(markeredgecolor='gray', marker='+'))

# labels
for plot in plots:
    plot.tick_params(axis='both', which='major', labelsize=7)
    plot.set_ylabel('Root Squared Error')
    plot.set_xlabel('Label')

plt.savefig('raw_data_en.pdf', format='pdf')
#plt.show()
