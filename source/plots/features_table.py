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
df = df[df['data_source'].str.contains('bild')]

# prepare data
df = df[['feature_generator', 'regressor_name', 'error']]
df['feature_generator'] = df['feature_generator'].str.replace('_', ' ').str.title()
df = df.sort_values('error')

print(df.to_latex(index=False, float_format='%11.3f'))
