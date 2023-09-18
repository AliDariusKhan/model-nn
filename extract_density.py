import os
import gemmi
import numpy as np

N_GRID = 8
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

def density_map_grid(residue, density_map, spacing=1.0):
    origin, x, y, z = get_grid_basis(residue)
    grid_corner = origin - ((N_GRID - 1) * spacing / 2) * (x + y + z)

    density_map_grid_values = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.float32)
    
    transform = gemmi.Transform()
    transform.mat.fromlist(np.column_stack([x, y, z]))
    transform.vec.fromlist(grid_corner)

    density_map.interpolate_values(density_map_grid_values, transform)
    return density_map_grid_values

def residue_density_map_generator(pdb_id):
    mtz = gemmi.read_mtz_file(os.path.join('modelcraft_outputs', pdb_id, 'modelcraft.mtz'))
    density_maps = [mtz.transform_f_phi_to_map(amplitude, phase) for amplitude, phase in MAPS]
    for density_map in density_maps:
        density_map.normalize()

    model_structure_path = os.path.join('modelcraft_outputs', f"{pdb_id}", "modelcraft.cif")
    model_structure = gemmi.read_structure(model_structure_path)

    for chain in model_structure[0]:
        for residue in chain:
            if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                continue

            all_map_values = []
            for density_map in density_maps:
                map_values = density_map_grid(residue, density_map).reshape(*3*N_GRID, 1)
                all_map_values.append(map_values)

            yield np.concatenate(all_map_values, axis=-1)

def main():
    modelcraft_folder = 'modelcraft_outputs'
    pdb_ids = []

    for root, _, files in os.walk(modelcraft_folder):
        if any(file.endswith('.cif') for file in files):
            pdb_ids.append(root)

    for pdb_id in pdb_ids:
        try:
            residue_density_map_generator(pdb_id)
        except:
            print(f"Failed to generate modelcraft for {pdb_id}")

if __name__ == "__main__":
    main()