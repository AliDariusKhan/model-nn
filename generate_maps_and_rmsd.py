import random
import requests
import os
import subprocess
from Bio import PDB
import gemmi
import numpy as np
import tensorflow as tf

N_GRID = 8

def fetch_mtz(pdb_id, output_dir='mtzs_for_modelcraft'):
    mtz_path = os.path.join(output_dir, f"{pdb_id}.mtz")
    if os.path.exists(mtz_path):
        print(f"MTZ for {pdb_id} already exists. Skipping download.")
        return

    base_url = "https://edmaps.rcsb.org/coefficients"
    mtz_url = f"{base_url}/{pdb_id}.mtz"
    response = requests.get(mtz_url, stream=True)
    response.raise_for_status()
    os.makedirs(output_dir, exist_ok=True)
    with open(mtz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def fetch_pdb(pdb_id, output_dir='pdb_files'):
    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        print(f"PDB for {pdb_id} already exists. Skipping download.")
        return

    base_url = "https://files.rcsb.org/download"
    pdb_url = f"{base_url}/{pdb_id}.pdb"
    response = requests.get(pdb_url, stream=True)
    response.raise_for_status()
    os.makedirs(output_dir, exist_ok=True)
    with open(pdb_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def generate_contents_json(pdb_id, output_dir='contents_files'):
    contents_json_path = os.path.join(output_dir, f"{pdb_id}_contents.json")
    if os.path.exists(contents_json_path):
        print(f"Contents JSON for {pdb_id} already exists. Skipping generation.")
        return

    cmd = ["modelcraft-contents", pdb_id, contents_json_path]
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to generate contents JSON for {pdb_id}: {result.stderr}")

def execute_modelcraft(pdb_id):
    directory_path = os.path.join('modelcraft_outputs', pdb_id)
    expected_files = [
        os.path.join(directory_path, "modelcraft.cif"),
        os.path.join(directory_path, "modelcraft.json"),
        os.path.join(directory_path, "modelcraft.mtz")
    ]

    if all(os.path.exists(file_path) for file_path in expected_files):
        print(f"Skipped modelcraft for {pdb_id}: files already present")
        return

    contents_json_path = os.path.join('contents_files', f"{pdb_id}_contents.json")
    data_path = os.path.join('mtzs_for_modelcraft', f"{pdb_id}.mtz")
    cmd = [
        "modelcraft", "xray",
        "--contents", contents_json_path,
        "--data", data_path,
        "--overwrite-directory",
        "--cycles", "1",
        "--directory", directory_path,
        "--phases", "PHIC,FOM"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to execute modelcraft for {pdb_id}: {result.stderr}")

def get_maps_and_distances(pdb_id, spacing=1.0):    
    mtz = gemmi.read_mtz_file(os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.mtz'))
    grid_fwt_phwt = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid_delfwt_phdelwt = mtz.transform_f_phi_to_map("DELFWT", "PHDELWT")
    grid_fwt_phwt.normalize()
    grid_delfwt_phdelwt.normalize()

    ref_structure_path = os.path.join('pdb_files', f"{pdb_id}.pdb")
    modelcraft_structure_path = os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.cif')

    ref_structure = gemmi.read_structure(ref_structure_path)
    modelcraft_structure = gemmi.read_structure(modelcraft_structure_path)

    for model_chain in modelcraft_structure[0]:
        for model_residue in model_chain:
            model_CA = model_residue.find_atom("CA", "\0")
            if model_CA:
                offset = (N_GRID - 1) * spacing / 2
                fwt_phwt_values = np.zeros((N_GRID, N_GRID, N_GRID))
                delfwt_phdelwt_values = np.zeros((N_GRID, N_GRID, N_GRID))
                
                for dx in range(N_GRID):
                    for dy in range(N_GRID):
                        for dz in range(N_GRID):
                            x = model_CA.pos.x + spacing * dx - offset
                            y = model_CA.pos.y + spacing * dy - offset
                            z = model_CA.pos.z + spacing * dz - offset
                            pos = gemmi.Position(x, y, z)
                            fwt_phwt_values[dx, dy, dz] = grid_fwt_phwt.interpolate_value(pos)
                            delfwt_phdelwt_values[dx, dy, dz] = grid_delfwt_phdelwt.interpolate_value(pos)

                min_distance = float('inf')
                nearest_ref_CA = None
                for ref_chain in ref_structure[0]:
                    for ref_residue in ref_chain:
                        ref_CA = ref_residue.find_atom("CA", "\0")
                        if ref_CA:
                            distance = model_CA.pos.dist(ref_CA.pos)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_ref_CA = ref_CA

                if nearest_ref_CA:
                    return np.concatenate([fwt_phwt_values.reshape(N_GRID, N_GRID, N_GRID, 1), delfwt_phdelwt_values.reshape(N_GRID, N_GRID, N_GRID, 1)], axis=-1), np.array([min_distance])

def create_model():
    x = inputs = tf.keras.Input(shape=(N_GRID, N_GRID, N_GRID, 2))
    _downsampling_args = {
        "padding": "same",
        "use_bias": False,
        "kernel_size": 3,
        "strides": 1,
    }

    filter_list = [32, 64, 128]
    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.Dense(8)(x)
    output = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=output)

def train():
    train_gen = generate_dataset("train")
    test_gen = generate_dataset("test")
    input = tf.TensorSpec(shape=(N_GRID, N_GRID, N_GRID, 2), dtype=tf.float32)
    output = tf.TensorSpec(shape=(1), dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_gen, output_signature=(input, output)
    )

    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = "test_1"

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)
    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    model = create_model()
    model.summary()

    model.compile(optimizer="adam", loss="mse",  metrics=['mse'])

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
        # reduce_lr_on_plat,
        # TqdmCallback(verbose=2),
        tensorboard_callback,
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

def generate_dataset(train_or_test):
    with open(f'./{train_or_test}_pdb_ids.txt', 'r') as f:
        pdb_ids = f.read().splitlines()
    for pdb_id in pdb_ids:
        yield get_maps_and_distances(pdb_id)

def generate_inputs(train_test_split=0.8):
    with open('./input_pdb_list.txt', 'r') as f:
        pdb_ids = f.read().splitlines()
    
    random.shuffle(pdb_ids)
    num_train = int(len(pdb_ids) * train_test_split)
    if num_train == 0:
        num_train = 1
    elif num_train == len(pdb_ids):
        num_train = len(pdb_ids) - 1

    train_ids = pdb_ids[:num_train]
    test_ids = pdb_ids[num_train:]

    for train_or_test, pdb_ids in [("train", train_ids), ("test", test_ids)]:
        for pdb_id in pdb_ids:
            with open(f'{train_or_test}_pdb_ids.txt', 'w') as file:
                file.write(f"{pdb_id}\n")
            fetch_pdb(pdb_id)
            fetch_mtz(pdb_id)
            generate_contents_json(pdb_id)
            execute_modelcraft(pdb_id)

def main():
    generate_inputs()
    train()

if __name__ == "__main__":
    main()