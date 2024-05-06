import numpy as np
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical

# Data from: https://www.cs.toronto.edu/~kriz/cifar.html

# unpickle function
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load data
x_train = []
y_train = []
for i in range(1, 5):
    data = unpickle(f"Assignment3\data_batch_{i}")
    x_train.append(data[b'data'])
    y_train.append(data[b'labels'])
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)
data = unpickle('Assignment3/test_batch')
x_test = data[b'data']
y_test = np.array(data[b'labels'])


# Reshape to (32,32,3)
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Categorize
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#Define model
#Input – (CNN – Relu – pooling)1 – (CNN – Relu – Pooling)2 – (CNN – Relu - CNN – Relu – Pooling)3 – (FNN )4 – (Softmax)5
model = Sequential()
model.add(Conv2D(64, (7, 7), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu',  padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',  padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Track training time and errors
train_errors = []
test_errors = []
train_time = []


# Train model
num_epochs = 10
batch_size = 64
for epoch in range(num_epochs):
    
    # Measure start time
    start_time = time.time()
    
    # Begin Traininig
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
    
    # Measure end time and store total time spent on single epoch
    end_time = time.time()
    epoch_time = end_time - start_time
    train_time.append(epoch_time)

    # Evaluate on training set
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    train_error = 1 - train_acc
    train_errors.append(train_error)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    test_error = 1 - test_acc
    test_errors.append(test_error)

    #print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.3f}s, Accuracy: {test_acc:.3f}, Test Error: {test_error: .3f},Train Error: {train_error:.3f}")

# The training time as function of epoch (10 data points)
epochs = range(1, num_epochs+1)
plt.plot(epochs, train_time)
plt.xlabel('Epoch')
plt.ylabel('Training Time (s)')
plt.title('Training Time as Function of Epoch')
plt.show()

# The training and testing errors after each epoch of training (20 data points in a single plot).
plt.plot(epochs, train_errors, label='Training Error')
plt.plot(epochs, test_errors, label='Testing Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and Testing Errors')
plt.legend()
plt.show()

# end of program
