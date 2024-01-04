import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from imblearn.over_sampling import SMOTE

#checking the data
val_x = pd.read_excel('pre-svm.xlsx')
train = pd.read_csv('pre-svm.csv')
X_train = train.sen
y_train = train.bis
X_train = pd.DataFrame()
val = val_x.SEn
X_train['id'] = train.id
X_train['SEn'] = train.SEn
val['bis'] = val_x.bis
X_1 = val[val['bis'] == 1] 


#y = np_utils.to_categorical(y)
#y_test = np_utils.to_categorical(val)
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_test = sc.transform(X_test)
X_val = sc.transform(X_val)
X_train=np.array(X_train)
X_train = X_train.reshape(-1, 1)
X_train = sc.fit_transform(X_train)
X_res, Y_res = SMOTE(random_state = 42).fit_resample(X_train, y_train)
X_res = pd.DataFrame(np.array(X_res))
Y_res = pd.DataFrame(np.array(Y_res))
X = X_res
X = pd.concat([X, Y_res], axis=1)
Y_res = tf.argmax(Y_res, axis=1)
Y_res = np.asarray(Y_res).astype('int')
Y_res = pd.DataFrame(np.array(Y_res))
X = pd.concat([X, Y_res], axis=1)
X.to_csv("smote.csv") 
########lof#########.
from sklearn.neighbors import LocalOutlierFactor
# identify outliers in the training dataset
lof = LocalOutlierFactor(n_neighbors=109 )
yhat = lof.fit_predict(X)
# select all rows that are not outliers
clf = LocalOutlierFactor()
b = clf.fit_predict(all)
mask_all = X[yhat != -1] 
delete_number = X[yhat == -1] 
outlier =clf.negative_outlier_factor_
mask_all.to_csv("lof-109.csv") 



#X = sc.transform(X)
####### normal Classifier###########
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
model = Sequential([
    tf.keras.Input(shape=(1,)),
    Dense(64, activation='relu'),# 8個隐藏神经元的全连接层
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.5),
    Dense(5, activation='softmax')])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_res, Y_res, batch_size = 64, epochs = 20, validation_data =(X_val, y_val))
loss, accuracy = model.evaluate(X_val, y_val)
model.summary()

# ------ Plot loss -----------#
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# ------ Plot accuracy -----------#
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
test = pd.read_csv('test_1.csv')
X_test = test.eeg
y_test = test.da
y_test = np_utils.to_categorical(y_test)
pred = model.predict(X_test)
#pred = model.predict(X_val)
#X_pred = model.predict(X)
#np.max(X_pred, axis=1)
y_pred_1 = tf.argmax(pred, axis=1)
y_test = tf.argmax(y_test, axis=1)
y_test = np.asarray(y_test).astype('float32')
y_pred_1 = np.asarray(y_pred_1).astype('float32')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_1)
all_cm = sum(sum(cm))
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/all_cm
acc

# save model
model.save('50_patient.h5')

