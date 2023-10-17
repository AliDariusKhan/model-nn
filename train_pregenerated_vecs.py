import argparse
import os
import random
import numpy as np
import tensorflow as tf
import math

def read_N_GRID(file_path):
    with open(file_path, 'r') as f:
        return int(f.readline().strip())

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

def res_to_target(target_path):
    with open(target_path, 'r') as f:
        target_lines = f.readlines()

    res_to_target_dict = dict()
    for i in range(1, len(target_lines), 2):
        residue_identifier = target_lines[i]
        res_to_target_dict[residue_identifier] = np.array(target_lines[i + 1].strip().split(','), dtype=float)
    return res_to_target_dict

def data_generator(res_ids, all_map_values_path, res_to_target_dict, N_GRID):
    print("before res set")
    res_set = set(res_ids)
    print("before open map")
    with open(all_map_values_path, 'r') as map_values_file:
        print("after open map")
        lines = iter(map_values_file)
        next(lines)  # Skip the N_GRID line
        for line in lines:
            residue_id = line.strip()
            if residue_id not in res_set:
                continue
            fwt_phwt = next(lines)
            fwt_phwt_values = next(lines).strip().split(',')
            fwt_phwt_map_values = np.array(fwt_phwt_values, dtype=float).reshape(N_GRID, N_GRID, N_GRID, 1)

            delfwt_phdelwt = next(lines)
            delfwt_phdelwt_values = next(lines).strip().split(',')
            delfwt_phdelwt_map_values = np.array(delfwt_phdelwt_values, dtype=float).reshape(N_GRID, N_GRID, N_GRID, 1)

            all_map_values = np.concatenate([fwt_phwt_map_values, delfwt_phdelwt_map_values], axis=-1)
            target = res_to_target_dict[residue_id]

            yield all_map_values, target

def train(train_res_ids, test_res_ids, res_to_target_dict, all_map_values_path, N_GRID):
    print("before data_generator")
    train_gen = data_generator(train_res_ids, all_map_values_path, res_to_target_dict, N_GRID)
    test_gen = data_generator(test_res_ids, all_map_values_path, res_to_target_dict, N_GRID)
    input = tf.TensorSpec(shape=(N_GRID, N_GRID, N_GRID, 2), dtype=tf.float32)
    output = tf.TensorSpec(shape=(3), dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_gen, output_signature=(input, output)
    )

    num = 0
    while os.path.exists(f"models/model_{num}.best.hdf5"):
        num += 1

    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = f"model_{num}"

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)
    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    model = create_model(N_GRID)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="mse",  metrics=['mse'])

    logger = tf.keras.callbacks.CSVLogger(f"train_{name}.csv", append=True)
    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{name}", histogram_freq=1, profile_batch=(10, 30)
    )
    
    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        # tensorboard_callback,
        logger,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1,
        use_multiprocessing=True,
    )

def main():
    parser = argparse.ArgumentParser(description='Process input data folder.')
    parser.add_argument('--all_map_values', required=True, help='Path to the all map values file.')
    parser.add_argument('--target', required=True, help='Path to the target file.')
    args = parser.parse_args()

    all_map_values_path = args.all_map_values
    target_path = args.target

    N_GRID = read_N_GRID(all_map_values_path)
    print("N_GRID", N_GRID)
    res_to_target_dict = res_to_target(target_path)
    print("after target dict")

    # Splitting the data into training and testing sets
    residue_ids = list(res_to_target_dict.keys())
    random.shuffle(residue_ids)
    split_idx = int(len(residue_ids) * 0.8)
    train_res_ids, test_res_ids = residue_ids[:split_idx], residue_ids[split_idx:]
    print("train test split")
    train(train_res_ids, test_res_ids, res_to_target_dict, all_map_values_path, N_GRID)

if __name__ == "__main__":
    main()
