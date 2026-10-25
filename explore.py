import pandas as pd
from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()

df=pd.DataFrame(data.data,columns=data.feature_names)

df["Price"]=data.target
print(df.shape)
print(df.columns)
print(df.describe)