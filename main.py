from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
import numpy as np

#DataSet Manipulation
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')

X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

X_train = X_train.reshape(len(X_train), 100, 100, 3)/255.0
X_test = X_test.reshape(len(X_test), 100, 100, 3)/255.0
Y_train = Y_train.reshape(len(Y_train), 1)
Y_test = Y_test.reshape(len(Y_test), 1)

#Model Structure
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])


#Compiling Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting the Model
model.fit(X_train, Y_train, epochs=10, batch_size=64)

#Evaluation
model.evaluate(X_test, Y_test)


