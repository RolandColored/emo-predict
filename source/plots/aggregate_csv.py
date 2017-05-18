import glob
import os

import pandas as pd

folder = 'bow'
path = '../../results/' + folder

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, '*.csv'))))
df.to_csv(path + '/../' + folder + '.csv')
