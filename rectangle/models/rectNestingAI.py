# univariate bidirectional lstm example
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dropout, Reshape
import torch
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
print("gpus: ", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')


tf.config.set_visible_devices([], 'GPU')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Use the Neural Engine
if device.type == 'cpu':
    # Specify device as 'xnnpack' for Neural Engine
    device = torch.device('xnnpack')
elif device.type == 'cuda':
    # Use CUDA device
    torch.cuda.set_device(torch.cuda.current_device())"""

def create_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(10 * 3, activation='linear')(x)  # Output layer for position and rotation of 10 rectangles
    output_layer = layers.Reshape((10, 3))(output_layer)  # Reshape output to (None, 10, 3)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

#adding regularization
def create_model2(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.2),  # Adding dropout regularization
        Dense(64, activation='relu'),
        Dropout(0.2),  # Adding dropout regularization
        Dense(10 * 3, activation='linear'),
        Reshape((10, 3))
    ])
    return model

def create_model3(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(input_layer)  # Increased number of nodes to 128
    x = layers.Dense(128, activation='relu')(x)            # Additional hidden layer with 128 nodes
    x = layers.Dense(64, activation='relu')(x)             # Adding another hidden layer with 64 nodes
    output_layer = layers.Dense(10 * 3, activation='linear')(x)
    output_layer = layers.Reshape((10, 3))(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def create_model4(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(10 * 3, activation='linear'),
        layers.Reshape((10, 3))
    ])
    return model

def create_model5(input_shape):
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

    with open("/Users/raphael/Desktop/FYP/rectangularNesting.csv", 'r') as csvfile:
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
                x_data.append((int(row[3]), int(row[4])))
            else:
                angle = int(row[5])
                rotated = angle != 0
                y_data.append((int(row[1]), int(row[2]), int(row[3]), int(row[4]), rotated))
    return X, Y

def augment_data(X, Y):
    # Add data augmentation logic here (e.g., rotation, translation, scaling)
    return augmented_X, augmented_Y

def custom_loss(y_true, y_pred):
    # Define custom loss function here
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

n_steps = 3

X, Y = processCSV()
X = np.array(X)
Y = np.array(Y)

n_features = X.shape[2]

model = create_model3((22,))

#decaying learning rate
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * 0.95 ** epoch)

#reducing the learning rate for an increase in accuracy
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Using Adam optimizer with a dynamic learning rate
#model.compile(optimizer=optimizer, loss='mse')
model.compile(optimizer='adam', loss='mse')
#model.summary()

X_flat = X.reshape(X.shape[0], -1)
Y_mod = Y[:, :, [0, 1, 4]]  # Keep only the first, second, and fifth values (x-pos, y-pos, angle)

batch_size = 32

dataset = tf.data.Dataset.from_tensor_slices((X_flat, Y_mod))
dataset = dataset.shuffle(buffer_size=len(X_flat)).batch(batch_size)

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=len(X))

train_size = int(0.8 * len(X))
X_train, X_val = X_flat[:train_size], X_flat[train_size:]
Y_train, Y_val = Y_mod[:train_size], Y_mod[train_size:]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size=len(X_val)).batch(batch_size)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_dataset, epochs=100, batch_size=batch_size, validation_data=val_dataset, callbacks=[lr_scheduler])
#history = model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


#model.fit(X_flat, Y_mod, epochs=10, batch_size=32, validation_split=0.2)

model.summary()

# Save the model
model.save("tenRect.h5")
model.save("tenRect.keras")

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
print("Validation Loss:", loss)