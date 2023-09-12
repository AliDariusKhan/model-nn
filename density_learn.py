import random
import requests
import os
import subprocess
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

def fetch_pdb(pdb_id, output_dir='pdb_files', if_jarvis=True):
    if os.path.exists(f'/old_vault/pdb/pdb{pdb_id}.ent'):
        return
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

def align_with_csymmatch(pdb_id):
    if os.path.exists(f'/old_vault/pdb/pdb{pdb_id}.ent'):
        pdb_path = f'/old_vault/pdb/pdb{pdb_id}.ent'
    else:
        pdb_path = os.path.join('pdb_files', f"{pdb_id}.pdb")
    modelcraft_pdb_path = os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.cif')
    aligned_output_dir = "aligned_pdb"
    aligned_pdb_path = os.path.join(aligned_output_dir, f"{pdb_id}.pdb")
    
    os.makedirs(aligned_output_dir, exist_ok=True)
    
    cmd = [
        "csymmatch",
        "-pdbin-ref", modelcraft_pdb_path,
        "-pdbin", pdb_path,
        "-pdbout", aligned_pdb_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to align pdb model for {pdb_id} with csymmatch: {result.stderr}")

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

def get_map_values(residue, grid_fwt_phwt, grid_delfwt_phdelwt, spacing=1.0, populate_individually=False):
    origin, x, y, z = get_grid_basis(residue)
    grid_corner = origin - ((N_GRID - 1) * spacing / 2) * (x + y + z)

    fwt_phwt_values = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.float32)
    delfwt_phdelwt_values = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.float32)
    
    if populate_individually:
        for i in range(N_GRID):
            for j in range(N_GRID):
                for k in range(N_GRID):
                    vector = grid_corner + spacing * (i * x + j * y + k * z)
                    pos = gemmi.Position(*vector)
                    fwt_phwt_values[i, j, k] = grid_fwt_phwt.interpolate_value(pos)
                    delfwt_phdelwt_values[i, j, k] = grid_delfwt_phdelwt.interpolate_value(pos)
    else:
        transform = gemmi.Transform()
        transform.mat.fromlist(np.column_stack([x, y, z]))
        transform.vec.fromlist(grid_corner)

        grid_fwt_phwt.interpolate_values(fwt_phwt_values, transform)
        grid_delfwt_phdelwt.interpolate_values(delfwt_phdelwt_values, transform)

    return fwt_phwt_values, delfwt_phdelwt_values

def standard_position(position, unit_cell):
    return gemmi.Position(
        position.x % unit_cell.a,
        position.y % unit_cell.b,
        position.z % unit_cell.c)

def get_maps_and_distances(pdb_id, use_transform=True):
    mtz = gemmi.read_mtz_file(os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.mtz'))
    grid_fwt_phwt = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid_delfwt_phdelwt = mtz.transform_f_phi_to_map("DELFWT", "PHDELWT")
    grid_fwt_phwt.normalize()
    grid_delfwt_phdelwt.normalize()

    ref_structure_path = os.path.join('aligned_pdb', f"{pdb_id}.pdb")    
    model_structure_path = os.path.join('modelcraft_outputs', f"{pdb_id}", "modelcraft.cif")

    ref_structure = gemmi.read_structure(ref_structure_path)
    model_structure = gemmi.read_structure(model_structure_path)

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
            
            fwt_phwt_values, delfwt_phdelwt_values = get_map_values(residue, grid_fwt_phwt, grid_delfwt_phdelwt)            

            model_CA = residue.find_atom("CA", "\0")
            ref_CA = neighbor_search.find_nearest_atom(model_CA.pos, radius=5)
            if not ref_CA:
                continue
            model_CA_pos = standard_position(model_CA.pos, model_structure.cell)
            ref_CA_pos = standard_position(ref_CA.pos, ref_structure.cell)
            model_to_ref = model_CA_pos - ref_CA_pos

            return np.concatenate([
                fwt_phwt_values.reshape(N_GRID, N_GRID, N_GRID, 1), 
                delfwt_phdelwt_values.reshape(N_GRID, N_GRID, N_GRID, 1)], axis=-1), np.array([model_to_ref.length()])

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

def generate_dataset(train_or_test):
    with open(f'./{train_or_test}_pdb_ids.txt', 'r') as f:
        pdb_ids = f.read().splitlines()
    for pdb_id in pdb_ids:
        yield get_maps_and_distances(pdb_id)

def generate_inputs(train_test_split=0.8):
    with open('./input_pdb_list.txt', 'r') as f:
        pdb_ids = f.read().lower().split(',')
    
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
            align_with_csymmatch(pdb_id)

def main():
    generate_inputs()
    train()

if __name__ == "__main__":
    main()