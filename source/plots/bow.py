import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, rc

# rendering options
rcParams.update({'figure.autolayout': True})
rc('font', **{'family': 'Palatino'})
plt.style.use('ggplot')

# plot
df = pd.read_csv('../../results/bow.csv', index_col='feature_generator')
df = df[df['data_source'].str.contains('foxnews')]

linestyles = ['-', '--', ':']
i = 0
for column in list(df):
    if '_error' in column:
        df[column].plot(label=column.replace('_error', ''), marker='.', linestyle=linestyles[int(i / 7)])
        i += 1

# legend
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
frame = lgd.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

# labels
plt.ylabel('RMSE')
plt.xlabel('Anzahl Dimensionen')
#plt.show()
plt.savefig('bow_en.pdf', format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

