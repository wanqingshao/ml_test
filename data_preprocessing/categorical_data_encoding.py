import pandas as pd
import numpy as np

df = pd.DataFrame([["green", "M", 10.1, "class2"],
                  ["red", "L", 13.5, "class1"],
                  ["blue", "XL", 15.3, "class2"]])
df.columns = ["color", "size", "price", "classlabel"]

## mapping ordinal features

size_mapping = {"M":1,
                "L":2,
                "XL":3}
df["size"] = df["size"].map(size_mapping)

inverse_size_mapping = {size_mapping[i]:i for i in size_mapping.keys()}
df["size"] = df["size"].map(inverse_size_mapping)

## encoding class label, making sure labels are integer

class_mapping = {label:idx for idx, label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)
inverse_class_mapping = {class_mapping[i]:i for i in class_mapping.keys()}
df["classlabel"] = df["classlabel"].map(inverse_class_mapping)

## LabelEncoder from sklearn

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df["classlabel"])
class_le.inverse_transform(y)

## Onehot encoder

from sklearn.preprocessing import OneHotEncoder
color_ohe = OneHotEncoder()
color_ohe.fit_transform(np.array(df["color"]).reshape(-1,1)).toarray()


pd.get_dummies(df.iloc[:,0:3],drop_first=True)