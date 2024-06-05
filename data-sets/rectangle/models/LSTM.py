import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import csv
import matplotlib.pyplot as plt

# Define the RNN model
def create_rnn_model():
    inputs = layers.Input(shape=(11, 2))
    x = layers.LSTM(128, return_sequences=True)(inputs) # LSTM layer with 64 units
    x = layers.LSTM(64)(x)
    outputs = layers.Reshape((10, 3))(layers.Dense(30, activation='linear')(x)) # Dense layer for output
    model = Model(inputs=inputs, outputs=outputs)
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

            if(test_number == 100000):
                return X, Y
            
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
                width = ((int(row[3])-1)/249)*(int(row[3])<600) + ((int(row[3])-600)/350)*(int(row[3])>=600)
                height = ((int(row[4])-1)/249)*(int(row[4])<600) + ((int(row[4])-600)/350)*(int(row[4])>=600)
                x_data.append((width, height))
            else:
                angle = int(row[5])
                rotated = angle != 0
                y_data.append((int(row[1]), int(row[2]), int(row[3]), int(row[4]), rotated))
    return X, Y

X, Y = processCSV()
X = np.array(X)
Y = np.array(Y)

n_features = X.shape[2]

# Create the model
model = create_rnn_model()

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * ((0.955*(epoch < 25)) + (0.98*(epoch >= 25))) ** epoch)

# Compile the model
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

history = model.fit(train_dataset, epochs=30, batch_size=batch_size, validation_data=val_dataset, callbacks=[lr_scheduler])

# Print the model summary
model.summary()

# Save the model
model.save("lstm.h5")
model.save("lstm.keras")

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