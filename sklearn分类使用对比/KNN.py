import numpy as np
import pandas as pd
from sklearn import neighbors, datasets

f = open('x_data.csv')
df = pd.read_csv(f)
x_data = np.array(df)

f = open('y_data.csv')
df = pd.read_csv(f)
y_data_temp = np.array(df)
y_data = np.zeros(1)
for temp in y_data_temp:
    # y_data.append(int(temp[0]))
    y_data = np.append(y_data, temp[0])

y_data = y_data[1:len(y_data)]

n_neighbors = 15
weights = 'distance' # also set as 'uniform'
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(x_data,y_data)

result = clf.predict([[7, 57600000, 2, 9020, 16548.48],
                      [11, 892685.14, 1, 2050, 769556.15],
                      [7, 263.95, 1, 3090, 225.6],
                      [11, 4288.93, 1, 2050, 3697.35],
                      [7, 32916, 1, 2080, 28375.86],
                      [7, 7504889.99, 1, 2020, 7286301.01]])

print(result)