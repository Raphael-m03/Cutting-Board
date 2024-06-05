from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dropout, Reshape, BatchNormalization
import torch
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

"""gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpu)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)"""


tf.config.set_visible_devices([], 'GPU')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model (input_shape):
    input_layer = Input(shape=input_shape)
    # Flatten the input to ensure compatibility with Dense layers
    flattened_input = Reshape((-1,))(input_layer)
    x = Dense(512, activation='relu')(flattened_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(10 * 3, activation='linear')(x)
    # Reshape the output to (None, 10, 3)
    output_layer = Reshape((10, 3))(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def processCSV():
    current_test = 0
    X, Y = list(), list()

    with open("/Users/raphael/Desktop/FYP/recNest2.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        x_data = []
        y_data = []

        for row in csvreader:
            test_number = int(row[0])
            
            if test_number != current_test:
                if len(y_data) != 10:
                    abnormal_test_number = current_test  # Found abnormal test number
                    print(abnormal_test_number)
                else:
                    X.append(x_data)
                    Y.append(y_data) 
                current_test = test_number
                x_data, y_data = [], []

            if(row[1] == '-'):
                #normalise data using min max
                #width = ((int(row[3])-1)/249)*(int(row[3])<600) + ((int(row[3])-600)/350)*(int(row[3])>=600)
                #height = ((int(row[4])-1)/249)*(int(row[4])<600) + ((int(row[4])-600)/350)*(int(row[4])>=600)
                x_data.append((int(row[3]), int(row[4])))
            else:
                angle = int(row[5])
                rotated = angle != 0
                y_data.append((int(row[1]), int(row[2]), int(row[3]), int(row[4]), rotated))
    return X, Y

def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

X, Y = processCSV()
X = np.array(X)
Y = np.array(Y)

n_features = X.shape[2]

model = create_model((11,2))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * ((0.955*(epoch < 25)) + (0.98*(epoch >= 25))) ** epoch)

model.compile(optimizer='adam', loss='mse')

Y_mod = Y[:, :, [0, 1, 4]]

batch_size = 16  

train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y_mod[:train_size], Y_mod[train_size:]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size=len(X_val)).batch(batch_size)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="/Users/raphael/Desktop/FYP/weight.weights.h5", save_weights_only=True, verbose=1)

# Train the model
history = model.fit(train_dataset, epochs=100, batch_size=batch_size, validation_data=val_dataset, callbacks=[lr_scheduler])
#history = model.fit(train_dataset, epochs=100, batch_size=batch_size, validation_data=val_dataset, callbacks=[lr_scheduler, cp_callback])

model.summary()

# Save the model
model.save("tenRect3.h5")
model.save("tenRect3.keras")

Y_pred = model.predict(X_val)

# Calculate the difference between actual and predicted values
difference = Y_val - Y_pred

# Plot actual vs predicted values for each dimension
for i in range(Y_val.shape[2]):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_val[:, :, i].flatten(), Y_pred[:, :, i].flatten(), label=f'Dimension {i+1}', color='blue')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title(f'Actual vs Predicted Values (Dimension {i+1})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the difference in loss
plt.figure(figsize=(8, 6))
plt.hist(difference.flatten(), bins=50, color='blue', alpha=0.7)
plt.xlabel('Difference in Loss')
plt.ylabel('Frequency')
plt.title('Distribution of Difference in Loss')
plt.grid(True)
plt.show()

loss = model.evaluate(X_val, Y_val)
print(custom_loss(Y_val, model.predict(X_val)))
print("Validation Loss:", loss)