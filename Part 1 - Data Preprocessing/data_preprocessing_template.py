import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load dataset
dataset = pd.read_csv('Data.csv')

#split independent(x) and dependent(y) variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#split train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""