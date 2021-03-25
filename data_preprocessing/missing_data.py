
import pandas as pd
from io import StringIO

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# Remove samples or features with missing values

df.isnull().sum() ## Default asix is 0, column-wise summary
df.isnull().sum(axis=1)
df.dropna()  ## drop rows
df.dropna(axis  =1 ) ## drop columns
df.dropna(how="all")
df.dropna(thresh=3)
df.dropna(thresh=4)

# Value imputation
from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy = "mean")
imr.fit_transform(df) ## mean was used as filling values for each feature
df.fillna(df.mean())