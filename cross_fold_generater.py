import numpy as np
import pandas as pd
import os


data_path = os.path.join(os.getcwd(), 'training2017', 'REFERENCE.csv')
data=pd.read_csv(data_path)
data_list=data.to_numpy()
np.random.shuffle(data_list)
data1=data_list[0:1705]
data2=data_list[1705:1705*2+1]
data3=data_list[1705*2+1:1705*3+2]
data4=data_list[1705*3+2:1705*4+2]
data5=data_list[1705*4+2:1705*5+2]

data = {'file_name': data1[:,0],
        'label': data1[:,1]}

df = pd.DataFrame(data)
df.to_csv('cv/cv0.csv', index=True)


data = {'file_name': data2[:,0],
        'label': data2[:,1]}
df = pd.DataFrame(data)
df.to_csv('cv/cv1.csv', index=True)


data = {'file_name': data3[:,0],
        'label': data3[:,1]}
df = pd.DataFrame(data)
df.to_csv('cv/cv2.csv', index=True)


data = {'file_name': data4[:,0],
        'label': data4[:,1]}
df = pd.DataFrame(data)
df.to_csv('cv/cv3.csv', index=True)


data = {'file_name': data5[:,0],
        'label': data5[:,1]}
df = pd.DataFrame(data)
df.to_csv('cv/cv4.csv', index=True)