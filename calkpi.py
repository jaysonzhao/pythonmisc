import csv
import numpy as np




#计算得分value 为实际值 ，expect 是期望值， direction 是方向，也可用于权值，负数为越小越好，正数为越大越好.输入可以是NP ARRAY
def calpoint(value, expect, direction):
    return (value - expect)*direction/expect

millinletexpect=np.array([1.1, 11, 15])
millinletdir=np.array([-0.4, -0.3, -0.3])
linespeedexpect=np.array([0.9, 1, 1])
linespeeddir=np.array([-1, 0, 0])
caspoollevelexpect=np.array([1.4, 2.0, 1.0])
caspoolleveldir=np.array([-0.4, -0.2, -0.4])


with open('jtdataset\\dataresult1212caster_pool_level.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    output = open('jtdataset\\dataresult1212caster_pool_levelMark.csv', 'a')
    for row in reader:
        try:
           arr = np.array([float(row['DFA']),float(row['violmax']),float(row['rollingvarmax'])])

           output.write(row['filename'] + ',' + row['column'] + ','+ str(np.sum(calpoint(arr, caspoollevelexpect, caspoolleveldir)))+ '\n')

        except ValueError:
            continue
    output.close()