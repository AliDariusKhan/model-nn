import argparse
import os
import numpy as np
import tensorflow as tf
import math

def create_model(N_GRID):
    x = inputs = tf.keras.Input(shape=(N_GRID, N_GRID, N_GRID, 2))
    _downsampling_args = {
        "padding": "same",
        "use_bias": False,
        "kernel_size": 3,
        "strides": 1,
    }

    filter_list = [4 * 2**num_filters for num_filters in range(int(math.log(N_GRID, 2)))]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Flatten()(x)
    for filter in reversed(filter_list):
        x = tf.keras.layers.Dense(filter, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=output)

def load_data(all_map_values_path, target_path):
    with open(all_map_values_path, 'r') as f:
        lines = f.readlines()
    N_GRID = int(lines[0].strip())  # Reading N_GRID from the file

    with open(target_path, 'r') as f:
        target_lines = f.readlines()
    TARGET_NAME = target_lines[0].strip()  # Reading TARGET_NAME from the file

    data = []
    targets = []
    identifiers = []
    for i in range(1, len(lines), 5):  # Starting from 1 to skip the first line
        identifier = lines[i].strip()
        # FIX ITERATION
        map_values = np.array([list(map(float, line.strip().split(','))) for line in lines[i+2:i+N_GRID*N_GRID+2]])
        data.append(map_values.reshape(N_GRID, N_GRID, N_GRID, 2))
        identifiers.append(identifier)
    
    for i in range(1, len(target_lines), 2):  # Starting from 1 to skip the first line
        target = float(target_lines[i+1].strip())
        targets.append(target)
    
    return np.array(data), np.array(targets), identifiers, N_GRID

def save_predictions(identifiers, predictions, filename):
    with open(filename, 'w') as f:
        for identifier, prediction in zip(identifiers, predictions):
            f.write(identifier + '\n')
            f.write(str(prediction[0]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process input data folder.')
    parser.add_argument('--all_map_values', required=True, help='Path to the all map values file.')
    parser.add_argument('--target', required=True, help='Path to the target file.')
    args = parser.parse_args()

    all_map_values_path = args.all_map_values
    target_path = args.target

    data, targets, identifiers, N_GRID = load_data(all_map_values_path, target_path)
    
    # Splitting the data into training and testing sets
    split_idx = int(len(data) * 0.8)
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_targets, test_targets = targets[:split_idx], targets[split_idx:]
    train_identifiers, test_identifiers = identifiers[:split_idx], identifiers[split_idx:]

    model = create_model(N_GRID)  # Passing N_GRID to create_model function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="mse", metrics=['mse'])

    model.fit(train_data, train_targets, epochs=10, batch_size=8, validation_data=(test_data, test_targets))

    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

    save_predictions(train_identifiers, train_predictions, 'train_predictions.txt')
    save_predictions(test_identifiers, test_predictions, 'test_predictions.txt')

if __name__ == "__main__":
    main()
