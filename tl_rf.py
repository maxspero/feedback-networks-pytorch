import numpy as np

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

all_features = np.load('extracted_features.npy')
all_labels = np.load('labels.npy')

rf = RandomForestClassifier(n_estimators=128)

val_separator = int(all_features.shape[1]*0.9)
all_train = all_features[:,0:val_separator,:]
all_val = all_features[:,val_separator:,:]
labels_train = all_labels[:val_separator]
labels_val = all_labels[val_separator:]

train2 = all_train[2]
print(train2.shape)
print(train2)
print(labels_train.shape)

rf.fit(train2, labels_train)
joblib.dump(rf, 'rf_tl.pkl') 
# lr = joblib.load('lr_tl.pkl')


#subset acc
for val_it in all_val:
  score = rf.score(val_it, labels_val)
  print(score)

for val_it in all_val:
  total = val_it.shape[0]
  correct = 0
  pred = rf.predict_proba(val_it)
  pred_a = np.argmax(pred, axis=0)
  pred2 = np.zeros((pred[0].shape[0], 20))
  for p in range(len(pred)):
    pred2[:,p] = pred[p][:,1]
  pred3 = np.argmax(pred2, axis=1)
  for p in range(pred3.size):
    if labels_val[p, pred3[p]] == 1:
      correct += 1
  print(float(correct)/total)

for val_it in all_val:
  total = val_it.shape[0]
  correct = 0
  pred = rf.predict(val_it)
  correct_labels = np.any((pred + labels_val > 1), axis=1)
  correct = np.sum(correct_labels)
  print(float(correct)/total)





