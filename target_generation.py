import argparse
import os
import gemmi
import requests
import clipper

TARGET_NAME = "molprobity"

def get_reference_structure(pdb_id, output_dir='pdb_files'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{pdb_id}.pdb")
    jarvis_path = f'/old_vault/pdb/pdb{pdb_id}.ent'
    if os.path.exists(jarvis_path):
        print(f"PDB for {pdb_id} found on Jarvis")
        path = jarvis_path
    elif os.path.exists(path):
        print(f"PDB for {pdb_id} already downloaded")
    else:
        base_url = "https://files.rcsb.org/download"
        pdb_url = f"{base_url}/{pdb_id}.pdb"
        try:
            response = requests.get(pdb_url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PDB ID {pdb_id} not found on RCSB. Skipping...")
                return None
            else:
                raise e
    return gemmi.read_structure(path)

def get_minimol_from_path(model_path):
    fpdb = clipper.MMDBfile()
    minimol = clipper.MiniMol()
    try:
        fpdb.read_file(model_path)
        fpdb.import_minimol(minimol)
    except Exception as exception:
        raise Exception('Failed to import model file') from exception
    return minimol

def get_minimol_seq_nums(minimol):
    seq_nums = { }
    for chain in minimol:
        chain_id = str(chain.id()).strip()
        seq_nums[chain_id] = [ ]
        for residue in chain:
            seq_num = int(residue.seqnum())
            seq_nums[chain_id].append(seq_num)
    return seq_nums

def get_molprobity_data(model_path, seq_nums, model_id=None, out_queue=None):
    try:
        from mmtbx.command_line import load_model_and_data
        from mmtbx.command_line.molprobity import get_master_phil
        from mmtbx.validation.molprobity import molprobity, molprobity_flags
    except (ImportError, ModuleNotFoundError):
        print('WARNING: Failed to import MolProbity; continuing without MolProbity analyses')
        return

    try:
        cmdline = load_model_and_data(
            args=[ f'pdb.file_name="{model_path}"', 'quiet=True' ],
            master_phil=get_master_phil(),
            require_data=False,
            process_pdb_file=True)
        validation = molprobity(model=cmdline.model)
    except Exception:
        print('WARNING: Failed to run MolProbity; continuing without MolProbity analyses')
        return

    molprobity_data = { }
    molprobity_data['model_wide'] = { }
    molprobity_data['model_wide']['summary'] = { 'cbeta_deviations' : validation.cbetadev.n_outliers,
                                                 'clashscore' : validation.clashscore(),
                                                 'ramachandran_outliers' : validation.rama_outliers(),
                                                 'ramachandran_favoured' : validation.rama_favored(),
                                                 'rms_bonds' : validation.rms_bonds(),
                                                 'rms_angles' : validation.rms_angles(),
                                                 'rotamer_outliers' : validation.rota_outliers(),
                                                 'molprobity_score' : validation.molprobity_score() }

    molprobity_data['model_wide']['details'] = { 'clash' : [ ],
                                                 'c-beta' : [ ],
                                                 'nqh_flips' : [ ],
                                                 'omega' : [ ],
                                                 'ramachandran' : [ ],
                                                 'rotamer' : [ ] }

    molprobity_results = { 'clash' : validation.clashes.results,
                           'c-beta' : validation.cbetadev.results,
                           'nqh_flips' : validation.nqh_flips.results,
                           'omega' : validation.omegalyze.results,
                           'ramachandran' : validation.ramalyze.results,
                           'rotamer' : validation.rotalyze.results }

    for chain_id, chain_seq_nums in seq_nums.items():
        molprobity_data[chain_id] = { }
        for seq_num in chain_seq_nums:
            molprobity_data[chain_id][seq_num] = { category : None for category in molprobity_results }
            molprobity_data[chain_id][seq_num]['clash'] = 2


    for category, results in molprobity_results.items():
        for result in results:
            if category == 'clash':
                for atom in result.atoms_info:
                    chain_id = atom.chain_id.strip()
                    seq_num = int(atom.resseq.strip())
                    if molprobity_data[chain_id][seq_num][category] > 0:
                        molprobity_data[chain_id][seq_num][category] -= 1
                details_line = [ ' '.join(a.id_str().split()) for a in result.atoms_info ] + [ result.overlap ]
                molprobity_data['model_wide']['details'][category].append(details_line)
                continue

            chain_id = result.chain_id.strip()
            seq_num = int(result.resseq.strip())
            if category in ('ramachandran', 'rotamer'):
                if result.score < 0.3:
                    molprobity_data[chain_id][seq_num][category] = 0
                elif result.score < 2.0:
                    molprobity_data[chain_id][seq_num][category] = 1
                else:
                    molprobity_data[chain_id][seq_num][category] = 2
            else:
                if result.outlier:
                    chain_id = result.chain_id.strip()
                    seq_num = int(result.resseq.strip())
                    molprobity_data[chain_id][seq_num][category] = 0

            if result.outlier:
                score = result.deviation if category == 'c-beta' else result.score
                details_line = [ result.chain_id.strip(), result.resid.strip(), result.resname.strip(), score ]
                molprobity_data['model_wide']['details'][category].append(details_line)

    if out_queue is not None:
        out_queue.put(('molprobity', model_id, molprobity_data))

    return molprobity_data

def target_generator(target_path, input_data_folder, pdb_ids, cutoff=10):
    for pdb_id in pdb_ids:
        cif_path = os.path.join(input_data_folder, pdb_id, f"refined_{pdb_id}.mmcif")
    
        model_structure = gemmi.read_structure(cif_path)
        if not model_structure: 
            continue

        if TARGET_NAME == "Ca_dist":
            try:
                ref_structure = get_reference_structure(pdb_id)
            except:
                continue
            if not ref_structure:
                continue
            neighbor_search = gemmi.NeighborSearch(ref_structure[0], ref_structure.cell, max_radius=5)
            for chain_idx, chain in enumerate(ref_structure[0]):
                for res_idx, res in enumerate(chain):
                    for atom_idx, atom in enumerate(res):
                        if atom.name == 'CA':
                            neighbor_search.add_atom(atom, chain_idx, res_idx, atom_idx)

        elif TARGET_NAME == "molprobity":
            minimol = get_minimol_from_path(cif_path)
            seq_nums = get_minimol_seq_nums(minimol)
            molprobity_data = get_molprobity_data(cif_path, seq_nums)

        for chain in model_structure[0]:
            molprobity_chain = molprobity_data[chain.name]
            for residue in chain:
                if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                    continue

                if TARGET_NAME == "Ca_dist":
                    model_CA = residue.find_atom("CA", "\0")
                    ref_CA = neighbor_search.find_nearest_atom(model_CA.pos)
                    if not ref_CA:
                        continue
                    model_CA_pos = standard_position(model_CA.pos, model_structure.cell)
                    ref_CA_pos = standard_position(ref_CA.pos(), ref_structure.cell)
                    model_to_ref = model_CA_pos - ref_CA_pos
                    target = model_to_ref.length()
                    if target > cutoff:
                        continue

                elif TARGET_NAME == "molprobity":
                    molprobity_residue = molprobity_chain[residue.seqid.num]
                    target = molprobity_residue['rotamer']

                with open(target_path, 'a') as f:
                    f.write(f"{pdb_id},{chain.name},{residue.name},{residue.seqid.num}\n")
                    f.write(str(target) + "\n")

def standard_position(position, unit_cell):
    return gemmi.Position(
        position.x % unit_cell.a,
        position.y % unit_cell.b,
        position.z % unit_cell.c)

def generate_pdb_ids_file(input_data_folder):
    modelcraft_folder = input_data_folder
    pdb_ids = []
    for root, _, files in os.walk(modelcraft_folder):
        if any(file.endswith(f'refined_{root.split("/")[1]}.mmcif') for file in files):
            pdb_ids.append(root.split('/')[1])

    return pdb_ids

def create_new_training_data_folder():
    base_name = "training_data_"
    counter = 0
    while True:
        folder_name = base_name + str(counter)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name, counter
        counter += 1

def main():
    parser = argparse.ArgumentParser(description='Process input data folder.')
    parser.add_argument('--input_data', required=True, help='Path to the data folder.')
    args = parser.parse_args()
    input_data_folder = args.input_data

    training_data_folder, num = create_new_training_data_folder()
    target_path = os.path.join(training_data_folder, "target.txt")
    with open(target_path, 'w') as f:
        f.write(TARGET_NAME + "\n")
    pdb_ids = generate_pdb_ids_file(input_data_folder)
    target_generator(target_path, input_data_folder, pdb_ids)

if __name__ == "__main__":
    main()
    