import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

"""
Missing values
"""

csv_data = '''A,B,C,D
1.0,2.0, 3.0, 4.0
5.0, 6.0,, 8.0
10.0, 11.0, 12.0,'''

df = pd.read_csv(StringIO(csv_data))

### drop rows/columns with NA
df.dropna(axis = 1) ## columns with NA will be removed
df.dropna(axis = 0) ## rows with NA will be removed
df.dropna(thresh =4, axis = 0) ## requires at least 4 non NA samples
df.dropna(how = "all") ## requires all values are NA

si = SimpleImputer(missing_values = np.nan, strategy = "mean")
si.fit_transform(df.values)

df.fillna(df.mean())


"""
Categorical data
"""

df = pd.DataFrame([["green", "M", 10.1, "class2"],
                   ["red", "L", 13.5, "class1"],
                   ["blue", "XL", 15.3, "class2"]])
df.columns = ["color", "size", "price", "label"]


size_mapping = {"XL":3, "L":2, "M":1}
df["size"] = df["size"].map(size_mapping)

inv_size_mapping = {v:k for k, v in size_mapping.items()}
df["size"].map(inv_size_mapping)

class_mapping = {label:idx for idx, label in enumerate(df["label"])}
df["label"].map(class_mapping)

class_le = LabelEncoder()
y = class_le.fit_transform(df["label"].values)

class_le.inverse_transform(y)

ohe = OneHotEncoder(drop = "first")

ohe.fit_transform(np.array(df.iloc[:,0]).reshape(-1, 1)).toarray()

X = df[["color", "size", "price"]].values
c_transf = ColumnTransformer([
    ("onehot", OneHotEncoder(), [0]),
    ("nothing", 'passthrough', [1,2])])

c_transf.fit_transform(X).astype(float)


pd.get_dummies(df[["price", "color", "size"]], drop_first = True)