import glob
import os

import pandas as pd

folder = 'bow_regressors'
path = '../../results/' + folder

dfs = []
for file in glob.glob(os.path.join(path, '*.csv')):
    df = pd.read_csv(file)
    dfs.append(df.tail(1))

pd.concat(dfs).to_csv(path + '/../' + folder + '.csv')
