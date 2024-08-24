import numpy as np

val_features = np.load('val_features.npy')
val_labels = np.load('val_labels.npy')
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
print('val_features:',val_features.shape,'\n')
print('val_labels:',val_labels.shape,'\n')
print('train_features:',train_features.shape,'\n')
print('train_labels:',train_labels.shape,'\n')

"""
val_features: (10000, 720) 

val_labels: (10000,) 

train_features: (50000, 720) 

train_labels: (50000,) 

"""

print(val_features[2])
print(val_features[2][3])
print(val_labels[2])
print(train_features[2])
print(train_features[2][3])
print(train_labels[49999])
"""
[0. 0. 0. ... 0. 0. 0.]
0.0
8
[0.        0.        0.        ... 0.        0.7021133 0.       ]
0.0
1
"""