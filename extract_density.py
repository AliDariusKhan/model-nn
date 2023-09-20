import os
import gemmi
import numpy as np
import random
import subprocess
import json
import tensorflow as tf
import requests

N_GRID = 32
MAPS = [("FWT", "PHWT"), ("DELFWT", "PHDELWT")]

def gemmi_position_to_np_array(gemmi_pos):
    return np.array([getattr(gemmi_pos, coord) for coord in ['x', 'y', 'z']])

def get_grid_basis(residue):
    origin = gemmi_position_to_np_array(residue.find_atom("CA", "\0").pos)
    plane_point_1 = gemmi_position_to_np_array(residue.find_atom("N", "\0").pos)
    plane_point_2 = gemmi_position_to_np_array(residue.find_atom("C", "\0").pos)
    x = plane_point_1 - origin
    x_norm = x / np.linalg.norm(x)
    plane_vector = plane_point_2 - origin
    z = np.cross(x, plane_vector)
    z_norm = z / np.linalg.norm(z)
    y_norm = np.cross(x_norm, z_norm)
    return origin, x_norm, y_norm, z_norm

def density_map_grid(residue, density_map, spacing=0.5):
    origin, x, y, z = get_grid_basis(residue)
    grid_corner = origin - ((N_GRID - 1) * spacing / 2) * (x + y + z)

    density_map_grid_values = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.float32)
    
    transform = gemmi.Transform()
    transform.mat.fromlist(spacing * np.column_stack([x, y, z]))
    transform.vec.fromlist(grid_corner)

    density_map.interpolate_values(density_map_grid_values, transform)
    return density_map_grid_values

def pdb_id_generator(train_or_test):
    with open(f'./{train_or_test}_pdb_ids.txt', 'r') as f:
        pdb_ids = f.read().splitlines()
    for pdb_id in pdb_ids:
        yield pdb_id

def get_tortoize_data(pdb_id, cif_path, pdb_dir="modelcraft_pdb"):
    os.makedirs(pdb_dir, exist_ok=True)
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")    
    subprocess.Popen(
            f'gemmi convert {cif_path} {pdb_path}', shell=True)

    tortoize_process = subprocess.Popen(
        f'tortoize {pdb_path}',
        shell=True,
        stdout=subprocess.PIPE)
    tortoize_output = tortoize_process.communicate()[0]
    tortoize_dict = json.loads(tortoize_output)
    residues = tortoize_dict["model"]["1"]["residues"]

    rama_z_data = dict()
    for res in residues:
        chain_rama_z_data = rama_z_data.setdefault(res['pdb']['strandID'], {})
        chain_rama_z_data[res['pdb']['seqNum']] = res['ramachandran']['z-score']
    
    return rama_z_data

def get_reference_structure(pdb_id, output_dir='pdb_files'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(jarvis_path := f'/old_vault/pdb/pdb{pdb_id}.ent'):
        print(f"PDB for {pdb_id} found on Jarvis")
        path = jarvis_path
    elif os.path.exists(path):
        print(f"PDB for {pdb_id} already downloaded")
    else:
        base_url = "https://files.rcsb.org/download"
        pdb_url = f"{base_url}/{pdb_id}.pdb"
        response = requests.get(pdb_url, stream=True)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)    
    return gemmi.read_structure(path)

def residue_density_map_generator(train_or_test):
    pdb_ids = pdb_id_generator(train_or_test)
    for pdb_id in pdb_ids:
        mtz = gemmi.read_mtz_file(os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.mtz'))
        density_maps = [mtz.transform_f_phi_to_map(amplitude, phase) for amplitude, phase in MAPS]

        for density_map in density_maps:
            density_map.normalize()

        cif_path = os.path.join('modelcraft_outputs', f"{pdb_id}", "modelcraft.cif")
        model_structure = gemmi.read_structure(cif_path)

        ref_structure = get_reference_structure(pdb_id)
        neighbor_search = gemmi.NeighborSearch(ref_structure[0], ref_structure.cell, max_radius=5)
        for chain_idx, chain in enumerate(ref_structure[0]):
            for res_idx, res in enumerate(chain):
                for atom_idx, atom in enumerate(res):
                    if atom.name == 'CA':
                        neighbor_search.add_atom(atom, chain_idx, res_idx, atom_idx)

        for chain in model_structure[0]:
            for residue in chain:
                if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                    continue

                all_map_values = []
                for density_map in density_maps:
                    map_values = density_map_grid(residue, density_map).reshape(N_GRID, N_GRID, N_GRID, 1)
                    all_map_values.append(map_values)

                model_CA = residue.find_atom("CA", "\0")
                ref_CA = neighbor_search.find_nearest_atom(model_CA.pos, radius=5)
                if not ref_CA:
                    continue
                model_CA_pos = standard_position(model_CA.pos, model_structure.cell)
                ref_CA_pos = standard_position(ref_CA.pos, ref_structure.cell)
                model_to_ref = model_CA_pos - ref_CA_pos

                yield np.concatenate(all_map_values, axis=-1), np.array([model_to_ref.length()])

def standard_position(position, unit_cell):
    return gemmi.Position(
        position.x % unit_cell.a,
        position.y % unit_cell.b,
        position.z % unit_cell.c)

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
    train_gen = residue_density_map_generator("train")
    test_gen = residue_density_map_generator("test")
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

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss="mse",  metrics=['mse'])

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

def train_test_split(train_proportion=0.8):
    modelcraft_folder = 'modelcraft_outputs'
    pdb_ids = []
    for root, _, files in os.walk(modelcraft_folder):
        if any(file.endswith('.cif') for file in files):
            pdb_ids.append(root.split('/')[1])

    random.shuffle(pdb_ids)
    num_train = int(len(pdb_ids) * train_proportion)
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

def main():
    train_test_split()
    train()

if __name__ == "__main__":
    main()
    