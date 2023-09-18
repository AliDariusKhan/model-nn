import os
import requests

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
    print(f"Successfully downloaded MTZ for {pdb_id}")


def main():
    with open('./input_pdb_list.txt', 'r') as f:
        pdb_ids = f.read().lower().split()

    for pdb_id in pdb_ids:
        try:
            fetch_mtz(pdb_id)
        except:
            print(f"Failed to download data for {pdb_id}")


if __name__ == "__main__":
    main()
