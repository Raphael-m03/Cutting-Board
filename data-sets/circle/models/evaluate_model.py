import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def manual_min_max_scaling(radii):
    min_radius = min(radii)
    max_radius = max(radii)

    if min_radius == max_radius:
        return [0.5] * len(radii)

    scaled_radii = []
    for radius in radii:
        if radius == min_radius:
            scaled_radius = 0.0
        elif radius == max_radius:
            scaled_radius = 1.0
        else:
            scaled_radius = (radius - min_radius) / (max_radius - min_radius)

        scaled_radii.append(scaled_radius)

    return scaled_radii

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def prepare_input_data(radii):
    radii = [r / 20 for r in radii]

    if len(radii) > 20:
        raise ValueError("The input list contains more than 20 radii.")
    elif len(radii) < 20:
        radii = radii + [-1] * (20 - len(radii))

    #radii_scaled = manual_min_max_scaling(radii)
    radii_scaled = np.array(radii).reshape(1, -1)
    return radii_scaled

def predict_coordinates(model, radii_scaled):
    predictions_scaled = model.predict(radii_scaled)
    predictions = predictions_scaled * 200
    coordinates = predictions.reshape(20, 2)
    return coordinates

def main():
    model_path = '/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/models/circle_nesting_model_1.h5'
    #radii = [20, 20, 19, 16, 16, 15, 14, 14, 13, 12, 9, 8, 6, 6, 6, 5, 4, 4, 2, 2]
    #radii = [18, 12]
    radii = [20, 20, 20, 19, 18, 18, 17, 16, 16, 14, 10, 8, 7, 7, 6, 5, 5, 4, 3, 2]

    model = load_model(model_path)

    radii_scaled = prepare_input_data(radii)

    print(radii_scaled)

    coordinates = predict_coordinates(model, radii_scaled)

    print("Predicted Coordinates (x, y) for each circle:")
    for i, (x, y) in enumerate(coordinates):
        print(f"Circle {i+1}: x={x:.2f}, y={y:.2f}")

if __name__ == "__main__":
    main()
