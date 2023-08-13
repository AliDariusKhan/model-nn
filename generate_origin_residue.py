"""
    Generate Origin Residue for transformations
    Jordan Dialpuri 13/08/23
"""

import gemmi


def get_origin_residue(path: str = "/vault/pdb/fj/pdb5fji.ent.gz") -> gemmi.Residue:
    """
    Get the first amino acid residue from specified PDB file path
    :return: residue (gemmi.Residue)
    """

    structure: gemmi.Structure = gemmi.read_structure(path)

    for chain in structure[0]:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info.is_amino_acid():
                return residue


def strip_sidechain(residue: gemmi.Residue) -> gemmi.Residue:
    """
    Strip all sidechain atoms from amino acid residue
    :param residue: residue to strip
    :return: stripped residue
    """

    allowed_atoms = ["CA", "C", "CB"]
    atom_names = [atom.name for atom in residue]
    for name in atom_names:
        if name not in allowed_atoms:
            residue.remove_atom(name, "\0")

    return residue


def save_residue(residue: gemmi.Residue, path: str):
    """
    Save residue to specified path
    :param residue: residue to save
    :param path: filepath to save to
    :return: None
    """
    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chain = gemmi.Chain("A")

    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)

    structure.write_pdb(path)


def main():
    origin_residue = get_origin_residue()
    stripped_residue = strip_sidechain(origin_residue)
    save_residue(stripped_residue, "data/origin_residue.pdb")


if __name__ == "__main__":
    main()
