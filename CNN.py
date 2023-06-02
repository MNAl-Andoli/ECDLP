from pyDeepInsight import ImageTransformer, LogScaler
import numpy as np
from tensorflow.keras import datasets, layers, models

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time
from sklearn.decomposition import PCA

# Stop GPU cuda, to avoid cuda error imcomatibale between keras library and cuda nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load dataset
x_train=np.load("x_train2.npy")

'''x_train=np.load("X_train.npy")
pca_x_train = PCA(n_components=250)
x_train = pca_x_train.fit_transform(x_train)'''

#x_train=np.load("x_train2.npy")
#===normalize x_train. z-score
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
#===
y_train=np.load("y_train11.npy")

# -------------------------------------

x_test=np.load("x_test2.npy")
'''x_test=np.load("X_test.npy")
pca_x_test = PCA(n_components=250)
x_test = pca_x_test.fit_transform(x_test)
'''
#====z_score normalization
scaler2=StandardScaler()
scaler2.fit(x_test)
x_test=scaler2.transform(x_test)
#====
y_test=np.load("y_test11.npy")

print(x_train.shape, x_test.shape)

all_data_train=np.concatenate((x_train, x_test), axis= 0)
all_data_y=np.concatenate((y_train, y_test), axis= 0)
all_data_train.shape
t1=time.time()

# normalize data
ln = LogScaler()
all_data_train=ln.fit_transform(all_data_train)
#x_test=ln.fit_transform(x_test)

# transform data to image form towork in CNN
it = ImageTransformer(feature_extractor='tsne', 
                      pixels=50, random_state=1701, 
                      n_jobs=-1)

all_data_train = it.fit_transform(all_data_train)
#x_test = it.fit_transform(x_test)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    all_data_train, all_data_y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Create the convolutional base
input_shape=X_train[0].shape

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=20, batch_size=256)
# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy training : %.2f' % (accuracy*100))

t2=time.time()

print ("time consumption : ", t2-t1)

pred = model.evaluate(X_test, y_test)
print('Accuracy test: %.2f' % (accuracy2*100))


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

yhat_classes = model.predict_classes(X_test, verbose=0)
#conf_matrix = confusion_matrix(y_true=y_test, y_pred=dt_pred)
print('Precision: %.3f' % precision_score(y_test, yhat_classes))
print('Recall: %.3f' % recall_score(y_test, yhat_classes))
print('F1 Score: %.3f' % f1_score(y_test, yhat_classes))
print('Accuracy: %.3f' % accuracy_score(y_test, yhat_classes))
