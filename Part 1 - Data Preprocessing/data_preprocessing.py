import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load dataset
dataset = pd.read_csv('Data.csv')

#split independent(x) and dependent(y) variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#replace missing data with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()

#return enconded column (each name are replaced by a number)
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#create a boolean(0 or 1) column to each country (1 when the row country is the 
#same of column country)
onehotenconder = OneHotEncoder(categorical_features=[0])
x = onehotenconder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)