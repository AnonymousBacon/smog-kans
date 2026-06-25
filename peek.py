import pandas as pd

path = 'data/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv'
pd.set_option('display.max_columns', None)

df = pd.read_csv(path, nrows = 5)
print(df.columns.to_list())
print(df.head())