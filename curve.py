from scipy.spatial.distance import pdist
import csv
import pandas as pd
import numpy as np
from dfa import dfa

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    counter=0

    arr = []
    for row in reader:
        try:
            arr.append(float(row['CS_MetBox_Copper_Temp']))
        except ValueError:
            continue
    print(arr)
#计算圆周曲率
for i in range(len(arr)-2):
    x = (i-(i-1), arr[i]-arr[i-1])
    y = (i+1-i, arr[i+1]-arr[i])
    d = 1-pdist([x, y], 'cosine')
    sin = np.sqrt(1-d**2)
    dis = np.sqrt((-2)**2 + (arr[i-1]-arr[i+1])**2)
    k = 2*sin/dis
    print(i,arr[i],k)