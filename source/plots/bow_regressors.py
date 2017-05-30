import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# data
df = pd.read_csv('../../results/bow_regressors.csv')
df = df[df['data_source'].str.contains('bild')]

linestyles = ['-', ':']
fig, ax = plt.subplots(figsize=(8, 6))

# iterate over groups
groups = df.groupby('regressor_name')
i = 0
for label, group in groups:
    group = group.sort_values('num_features')
    group.plot(ax=ax, x='num_features', y='error', label=label, marker='.',
               linestyle=linestyles[int(i / 6)], ylim=(0.09, 0.17))
    i += 1

# legend
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
frame = lgd.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

# labels
plt.ylabel('RMSE')
plt.xlabel('Anzahl LSI-Konzepte')
plt.xticks([100, 250, 500, 1000])
plt.savefig('bow_regressors_de.pdf', format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
