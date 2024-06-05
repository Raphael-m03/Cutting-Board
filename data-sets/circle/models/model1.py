import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape

def read_csv(filepath, nb_test_cases):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        test_cases = []
        current_test_case = []
        current_test_case_number = 0

        for row in reader:
            try:
                test_case_number = int(row[0].split("_")[-1])
            except (IndexError, ValueError):
                continue
            
            if test_case_number >= nb_test_cases + 1 and nb_test_cases != -1:
                break
            
            if(current_test_case_number == 20):
                test_cases.append(current_test_case)
                current_test_case = []
                current_test_case_number = 0

            radius = float(row[10])
            x = float(row[8])
            y = float(row[9])
            current_test_case.append([radius, x, y])
            current_test_case_number += 1

    return test_cases

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(40))
    #model.add(Reshape((20, 2)))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=10):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)
    return history

def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss}, Test MAE: {mae}")
    return loss, mae

def main():
    filepath = '/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/filtered_output.csv'
    nb_test_cases = -1
    
    test_cases = read_csv(filepath, nb_test_cases)
    
    X = []
    y = []

    for test_case in test_cases:
        radii = [circle[0] for circle in test_case]
        coords = [circle[1:] for circle in test_case]
        
        X.append(radii)
        y.append(np.array(coords).flatten())

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    model = build_model(input_dim=X_train.shape[1])

    train_model(model, X_train, y_train, epochs=50, batch_size=10)

    evaluate_model(model, X_test, y_test)

    predictions = model.predict(X_test)
    predictions = predictions.reshape(predictions.shape[0], -1).reshape(predictions.shape)

    model.summary()
    model.save('/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/models/circle_nesting_model_1.h5')

if __name__ == "__main__":
    main()