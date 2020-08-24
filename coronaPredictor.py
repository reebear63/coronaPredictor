import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
#from sklearn import

#### LOAD DATA ####
data = pd.read_csv('c:\\Users\\andreasg\\Documents\\Golgath\\Python\\corona\\coronaCases.csv', sep=',')
#data = data[['id', 'cases']]
print('-'*30)
print('HEAD') 
print('-'*30)
print (data.head())

#### PREPARE DATA ####
print('-'*30)
print('PREPARE DATA')
print('-'*30)
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)
plt.plot(y, '-m')
plt.show()

polyFeat = PolynomialFeatures(degree=2)
x = polyFeat.fit_transform(x)
print(x.head)