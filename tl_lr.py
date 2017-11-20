import numpy as np

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

all_features = np.load('extracted_features.npy')
all_labels = np.load('labels.npy')

lr = LogisticRegression()

val_separator = int(all_features.shape[1]*0.9)
all_train = all_features[:,0:val_separator,:]
all_val = all_features[:,val_separator:,:]
labels_train = all_labels[:val_separator]
labels_val = all_labels[val_separator:]

train2 = all_train[2]
print(train2.shape)
print(train2)
print(labels_train.shape)

lr.fit(train2, labels_train)
joblib.dump(lr, 'lr_tl.pkl') 
# lr = joblib.load('lr_tl.pkl')


for val_it in all_val:
  score = lr.score(val_it, labels_val)
  print(score)



