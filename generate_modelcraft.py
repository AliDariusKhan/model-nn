import os
import subprocess


def generate_contents_json(pdb_id, output_dir='contents_files'):
    contents_json_path = os.path.join(output_dir, f"{pdb_id}_contents.json")
    if os.path.exists(contents_json_path):
        print(f"Contents JSON for {pdb_id} already exists. Skipping generation.")
        return

    cmd = ["modelcraft-contents", pdb_id, contents_json_path]
    os.makedirs(output_dir, exist_ok=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        if process.returncode != 0:
            for line in process.stderr:
                print(line.strip())
            raise RuntimeError(f"Failed to generate contents JSON for {pdb_id}")

    if not os.path.exists(contents_json_path) or os.path.getsize(contents_json_path) == 0:
        raise RuntimeError(f"Failed to generate contents JSON for {pdb_id}: File not created or empty")

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
        "PHIC_ALL_LS,FOM",
        "PHIC,FOM",
        "PHIC_ALL,FOM",
        "PHWT,FOM",
        "PHDELWT,FOM",
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
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            if process.returncode != 0:
                for line in process.stderr:
                    print(line.strip())
                print(f"Failed to execute modelcraft for {pdb_id} with phases '{phase_arg}': {process.stderr.read().strip()}")

    for expected_file in expected_files:
        if not os.path.exists(expected_file) or os.path.getsize(expected_file) == 0:
            raise RuntimeError(f"Failed to execute modelcraft for {pdb_id}: Output file '{expected_file}' not created or empty")

def main():
    with open('successful_mtz_downloads.txt', 'r') as f:
        pdb_ids = f.read().splitlines()
    
    successful_pdb_ids = []

    for pdb_id in pdb_ids:
        try:
            generate_contents_json(pdb_id)
            execute_modelcraft(pdb_id)
            successful_pdb_ids.append(pdb_id)
        except:
            print(f"Failed to generate modelcraft for {pdb_id}")

    with open('successful_modelcraft_generations.txt', 'w') as f:
        for pdb_id in successful_pdb_ids:
            f.write(f"{pdb_id}\n")

if __name__ == "__main__":
    main()
