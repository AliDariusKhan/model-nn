import os
import requests
import subprocess

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

    phase_args = [
        "PHIC,FOM",
        "PHIC_ALL,FOM",
        "PHWT,FOM",
        "PHDELWT,FOM",
        "PHIC_ALL_LS,FOM",
    ]

    for phase_arg in phase_args:
        cmd = [
            "modelcraft", "xray",
            "--contents", contents_json_path,
            "--data", data_path,
            "--overwrite-directory",
            "--cycles", "1",
            "--directory", directory_path,
            "--basic",
            "--disable-sheetbend",
            "--disable-pruning",
            "--disable-parrot",
            "--disable-dummy-atoms",
            "--disable-waters",
            "--disable-side-chain-fixing",
            "--phases", phase_arg,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        else:
            print(f"Failed to execute modelcraft for {pdb_id} with phases '{phase_arg}': {result.stderr}")

    if result.returncode != 0:
        print(f"Failed to execute modelcraft for {pdb_id}: exhausted all phase options")

def main():
    with open('./input_pdb_list.txt', 'r') as f:
        pdb_ids = [pid.strip() for pid in f.read().lower().split(',')]

    for pdb_id in reversed(pdb_ids):
        fetch_mtz(pdb_id)
        generate_contents_json(pdb_id)
        execute_modelcraft(pdb_id)

if __name__ == "__main__":
    main()
