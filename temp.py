from pycaret.regression import *
import numpy as np

from pycaret.datasets import get_data
data = get_data('insurance')

s = setup(data, target = 'charges', session_id = 123)

df = models()

keys = df.index.to_list()
values = df['Name'].to_list()

res = {}
for key in keys:
    for value in values:
        res[key] = value
        values.remove(value)
        break

np.save('regr_dic.npy', res)