from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time

from sklearn.model_selection import train_test_split


# load dataset
X_train=np.load("x_train2.npy")
    
scaler=StandardScaler()
scaler.fit(X_train)
x_train=scaler.transform(X_train)
#===
Y_train=np.load("y_train11.npy")

# -------------------------------------

X_test=np.load("x_test2.npy")

#====z_score normalization
scaler2=StandardScaler()
scaler2.fit(X_test)
X_test=scaler2.transform(X_test)
#====
Y_test=np.load("y_test11.npy")

print(X_train.shape, X_test.shape)

t1= time.time()
input_layer= Input(shape=(X_train.shape[1],))
encoded = Dense(units=512, activation='relu')(input_layer)
encoded = Dense(units=256, activation='relu')(encoded)
encoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=256, activation='relu')(encoded)
decoded = Dense(units=512, activation='relu')(decoded)
decoded = Dense(units=X_train.shape[1], activation='sigmoid')(decoded)
autoencoder=Model(input_layer, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))



encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)
print(encoded_train.shape, encoded_test.shape)



from sklearn.neural_network import BernoulliRBM
import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

'''digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
Y = digits.target
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)'''

logistic = linear_model.LogisticRegression(C=100)
rbm1 = BernoulliRBM(n_components=100, learning_rate=0.06, n_iter=100, verbose=1, random_state=101)
rbm2 = BernoulliRBM(n_components=90, learning_rate=0.06, n_iter=3, verbose=1, random_state=101)
rbm3 = BernoulliRBM(n_components=60, learning_rate=0.06, n_iter=100, verbose=1, random_state=101)
#DBN3 = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])
DBN3 = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

DBN3.fit(encoded_train, Y_train)

print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        DBN3.predict(encoded_test))))
t2= time.time()

print("time consumption is : ", t2-t1)

