import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# plot
df = pd.read_csv('../../results/raw_data_en.csv')
df.boxplot(showmeans=True, flierprops=dict(markeredgecolor='gray', marker='+'))

# labels
plt.ylabel('RMSE')
plt.xlabel('Label')
plt.savefig('raw_data_en.pdf', format='pdf', bbox_inches='tight')
