import requests
import os
import subprocess
from Bio import PDB
import gemmi

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

def get_maps_and_rmsd(pdb_id):    
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
                fwt_phwt_value = grid_fwt_phwt.interpolate_value(model_CA.pos)
                delfwt_phdelwt_value = grid_delfwt_phdelwt.interpolate_value(model_CA.pos)

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
                    yield fwt_phwt_value, delfwt_phdelwt_value, min_distance

def main():
    with open('./input_pdb_id.txt', 'r') as f:
        pdb_ids = f.read().splitlines()

    for pdb_id in pdb_ids:
        try:
            fetch_pdb(pdb_id)
            fetch_mtz(pdb_id)
            generate_contents_json(pdb_id)
            execute_modelcraft(pdb_id)
            get_maps_and_rmsd(pdb_id)
        except requests.HTTPError:
            print(f"Failed to process files for {pdb_id}")

if __name__ == "__main__":
    main()