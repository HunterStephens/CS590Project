from openbabel import pybel
from jinja2 import Template
import numpy as np
import matplotlib.pyplot as plt
import biotite.structure as struc
import itertools
from numpy.linalg import norm
import biotite.sequence as seq
import biotite.structure.info as info
import biotite.structure.graphics as graphics

tmpl = Template("""
{{CH}} {{SP}}
{{XYZ}}
""")

# 'CA' is not in backbone,
# as we want to include the rotation between 'CA' and 'CB'
BACKBONE = ["N", "C", "O", "H"]

N_CA_LENGTH       = 1.46
CA_C_LENGTH       = 1.54
C_N_LENGTH        = 1.34
C_O_LENGTH        = 1.43
C_O_DOUBLE_LENGTH = 1.23
N_H_LENGTH        = 1.01
O_H_LENGTH        = 0.97

def pdb2psi4mol(fpath):

    mol = next(pybel.readfile("pdb", fpath))
    xyz = mol.write('xyz').split("\n")[2:]
    xyz = "\n".join(xyz)
    output = tmpl.render(CH=mol.charge,
                         SP=mol.spin,
                         XYZ=xyz)
    return output

def rotamer(residue, angle):

    bond_list = residue.bonds

    # --- Identify rotatable bonds ---
    rotatable_bonds = struc.find_rotatable_bonds(residue.bonds)

    # --- Do not rotate about backbone bonds ---
    for atom_name in BACKBONE:
        index = np.where(residue.atom_name == atom_name)[0][0]
        rotatable_bonds.remove_bonds_to(index)

    # --- VdW radii of each atom, required for the next step ---
    vdw_radii = np.zeros(residue.array_length())

    for i, element in enumerate(residue.element):
        vdw_radii[i] = info.vdw_radius_single(element)
    # --- The Minimum required distance between two atoms is mean of their VdW radii ---
    vdw_radii_mean = (vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]) / 2

    rotamer_coord = np.zeros((1, residue.array_length(), 3))

    # Coordinates for the current rotamer model
    coord = residue.coord.copy()
    for atom_i, atom_j, _ in rotatable_bonds.as_array():
        # The bond axis
        axis = coord[atom_j] - coord[atom_i]
        # Position of one of the involved atoms
        support = coord[atom_i]

        # Only atoms at one side of the rotatable bond should be moved
        # So the original Bondist is taken...
        bond_list_without_axis = residue.bonds.copy()
        # ...the rotatable bond is removed...
        bond_list_without_axis.remove_bond(atom_i, atom_j)
        # ...and these atoms are found by identifying the atoms that
        # are still connected to one of the two atoms involved
        rotated_atom_indices = struc.find_connected(
            bond_list_without_axis, root=atom_i
        )

        # Rotate
        coord[rotated_atom_indices] = struc.rotate_about_axis(
            coord[rotated_atom_indices], axis, angle, support
        )
        accepted = False
        while not accepted:
            # Rotate
            coord[rotated_atom_indices] = struc.rotate_about_axis(
                coord[rotated_atom_indices], axis, angle, support
            )

            # Check if the atoms clash with each other:
            # The distance between each pair of atoms must be larger
            # than the sum of their VdW radii, if they are not bonded to
            # each other
            accepted = True
            distances = struc.distance(
                coord[:, np.newaxis], coord[np.newaxis, :]
            )
            clashed = distances < vdw_radii_mean
            for clash_atom1, clash_atom2 in zip(*np.where(clashed)):
                if clash_atom1 == clash_atom2:
                    # Ignore distance of an atom to itself
                    continue
                if (clash_atom1, clash_atom2) not in bond_list:
                    # Nonbonded atoms clash
                    # -> structure is not accepted
                    accepted = False


    rotamer_coord[0] = coord
    rotamers = struc.from_template(residue, rotamer_coord)

    ### Superimpose backbone onto first model for better visualization ###
    rotamers, _ = struc.superimpose(
        rotamers[0], rotamers, atom_mask=struc.filter_backbone(rotamers)
    )

    return rotamers[0]

def calculate_atom_coord_by_z_rotation(coord1, coord2, angle, bond_length):
    rot_axis = [0, 0, 1]

    # Calculate the coordinates of a new atoms by rotating the previous
    # bond by the given angle (usually 120 degrees)
    new_coord = struc.rotate_about_axis(
            atoms = coord2,
            axis = rot_axis,
            angle = np.deg2rad(angle),
            support = coord1
        )

    # Scale bond to correct bond length
    bond_vector = new_coord - coord1
    new_coord = coord1 + bond_vector * bond_length / norm(bond_vector)

    return new_coord

def assemble_peptide(sequence):
    res_names = [seq.ProteinSequence.convert_letter_1to3(r) for r in sequence]
    peptide = struc.AtomArray(length=0)


    for res_id, res_name, connect_angle in zip(np.arange(1, len(res_names)+1),
                                               res_names,
                                               itertools.cycle([120, -120])):
        # Create backbone
        atom_n = struc.Atom(
            [0.0, 0.0, 0.0], atom_name="N", element="N"
        )

        atom_ca = struc.Atom(
            [0.0, N_CA_LENGTH, 0.0], atom_name="CA", element="C"
        )

        coord_c = calculate_atom_coord_by_z_rotation(
            atom_ca.coord, atom_n.coord, 120, CA_C_LENGTH
        )
        atom_c = struc.Atom(
            coord_c, atom_name="C", element="C"
        )

        coord_o = calculate_atom_coord_by_z_rotation(
            atom_c.coord, atom_ca.coord, 120, C_O_DOUBLE_LENGTH
        )
        atom_o = struc.Atom(
            coord_o, atom_name="O", element="O"
        )

        coord_h = calculate_atom_coord_by_z_rotation(
            atom_n.coord, atom_ca.coord, -120, N_H_LENGTH
        )
        atom_h = struc.Atom(
            coord_h, atom_name="H", element="H"
        )

        backbone = struc.array([atom_n, atom_ca, atom_c, atom_o, atom_h])
        backbone.res_id[:] = res_id
        backbone.res_name[:] = res_name

        # Add bonds between backbone atoms
        bonds = struc.BondList(backbone.array_length())
        bonds.add_bond(0, 1, struc.BondType.SINGLE) # N-CA
        bonds.add_bond(1, 2, struc.BondType.SINGLE) # CA-C
        bonds.add_bond(2, 3, struc.BondType.DOUBLE) # C-O
        bonds.add_bond(0, 4, struc.BondType.SINGLE) # N-H
        backbone.bonds = bonds


        # Get residue from dataset
        residue = info.residue(res_name)
        # Superimpose backbone of residue
        # with backbone created previously
        _, transformation = struc.superimpose(
            backbone[struc.filter_backbone(backbone)],
            residue[struc.filter_backbone(residue)]
        )
        residue = struc.superimpose_apply(residue, transformation)
        # Remove backbone atoms from residue because they are already
        # existing in the backbone created prevoisly
        side_chain = residue[~np.isin(
            residue.atom_name,
            ["N", "CA", "C", "O", "OXT", "H", "H2", "H3", "HXT"]
        )]


        # Assemble backbone with side chain (including HA)
        # and set annotation arrays
        residue = backbone + side_chain
        residue.bonds.add_bond(
            np.where(residue.atom_name == "CA")[0][0],
            np.where(residue.atom_name == "CB")[0][0],
            struc.BondType.SINGLE
        )
        residue.bonds.add_bond(
            np.where(residue.atom_name == "CA")[0][0],
            np.where(residue.atom_name == "HA")[0][0],
            struc.BondType.SINGLE
        )
        residue.chain_id[:] = "A"
        residue.res_id[:] = res_id
        residue.res_name[:] = res_name
        peptide += residue

        # Connect current residue to existing residues in the chain
        if res_id > 1:
            index_prev_ca = np.where(
                (peptide.res_id == res_id-1) &
                (peptide.atom_name == "CA")
            )[0][0]
            index_prev_c = np.where(
                (peptide.res_id == res_id-1) &
                (peptide.atom_name == "C")
            )[0][0]
            index_curr_n = np.where(
                (peptide.res_id == res_id) &
                (peptide.atom_name == "N")
            )[0][0]
            index_curr_c = np.where(
                (peptide.res_id == res_id) &
                (peptide.atom_name == "C")
            )[0][0]
            curr_residue_mask = peptide.res_id == res_id

            # Adjust geometry
            curr_coord_n  = calculate_atom_coord_by_z_rotation(
                peptide.coord[index_prev_c],  peptide.coord[index_prev_ca],
                connect_angle, C_N_LENGTH
            )
            peptide.coord[curr_residue_mask] -=  peptide.coord[index_curr_n]
            peptide.coord[curr_residue_mask] += curr_coord_n
            # Adjacent residues should show in opposing directions
            # -> rotate residues with even residue ID by 180 degrees
            if res_id % 2 == 0:
                coord_n = peptide.coord[index_curr_n]
                coord_c = peptide.coord[index_curr_c]
                peptide.coord[curr_residue_mask] = struc.rotate_about_axis(
                    atoms = peptide.coord[curr_residue_mask],
                    axis = coord_c - coord_n,
                    angle = np.deg2rad(180),
                    support = coord_n
                )

            # Add bond between previous C and current N
            peptide.bonds.add_bond(
                index_prev_c, index_curr_n, struc.BondType.SINGLE
            )


    # Add N-terminal hydrogen
    atom_n = peptide[(peptide.res_id == 1) & (peptide.atom_name == "N")][0]
    atom_h = peptide[(peptide.res_id == 1) & (peptide.atom_name == "H")][0]
    coord_h2 = calculate_atom_coord_by_z_rotation(
        atom_n.coord, atom_h.coord, -120, N_H_LENGTH
    )
    atom_h2 = struc.Atom(
        coord_h2,
        chain_id="A", res_id=1, res_name=atom_h.res_name, atom_name="H2",
        element="H"
    )
    peptide = struc.array([atom_h2]) + peptide
    peptide.bonds.add_bond(0, 1, struc.BondType.SINGLE) # H2-N

    # Add C-terminal hydroxyl group
    last_id = len(sequence)
    index_c = np.where(
        (peptide.res_id == last_id) & (peptide.atom_name == "C")
    )[0][0]
    index_o = np.where(
        (peptide.res_id == last_id) & (peptide.atom_name == "O")
    )[0][0]
    coord_oxt = calculate_atom_coord_by_z_rotation(
        peptide.coord[index_c], peptide.coord[index_o], connect_angle, C_O_LENGTH
    )
    coord_hxt = calculate_atom_coord_by_z_rotation(
        coord_oxt, peptide.coord[index_c], connect_angle, O_H_LENGTH
    )
    atom_oxt = struc.Atom(
        coord_oxt,
        chain_id="A", res_id=last_id, res_name=peptide[index_c].res_name,
        atom_name="OXT", element="O"
    )
    atom_hxt = struc.Atom(
        coord_hxt,
        chain_id="A", res_id=last_id, res_name=peptide[index_c].res_name,
        atom_name="HXT", element="H"
    )
    peptide = peptide + struc.array([atom_oxt, atom_hxt])
    peptide.bonds.add_bond(index_c, -2, struc.BondType.SINGLE) # C-OXT
    peptide.bonds.add_bond(-2,      -1, struc.BondType.SINGLE) # OXT-HXT


    return peptide

def rotate_residue(mol, res_id, bond_number, angle):
    # --- retrieve residue ---
    residue = mol[mol.res_id == res_id]
    bond_list = residue.bonds

    # --- Identify rotatable bonds ---
    rotatable_bonds = struc.find_rotatable_bonds(residue.bonds)

    # --- Do not rotate about backbone bonds ---
    for atom_name in BACKBONE:
        index = np.where(residue.atom_name == atom_name)[0][0]
        rotatable_bonds.remove_bonds_to(index)

    # --- VdW radii of each atom, required for the next step ---
    vdw_radii = np.zeros(residue.array_length())

    for i, element in enumerate(residue.element):
        vdw_radii[i] = info.vdw_radius_single(element)
    # --- The Minimum required distance between two atoms is mean of their VdW radii ---
    vdw_radii_mean = (vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]) / 2

    rotamer_coord = np.zeros((1, residue.array_length(), 3))

    # Coordinates for the current rotamer model
    coord = residue.coord.copy()
    atom_i, atom_j, _ = rotatable_bonds.as_array()[0]
    # The bond axis
    axis = coord[atom_j] - coord[atom_i]
    # Position of one of the involved atoms
    support = coord[atom_i]

    # Only atoms at one side of the rotatable bond should be moved
    # So the original Bondist is taken...
    bond_list_without_axis = residue.bonds.copy()
    # ...the rotatable bond is removed...
    bond_list_without_axis.remove_bond(atom_i, atom_j)
    # ...and these atoms are found by identifying the atoms that
    # are still connected to one of the two atoms involved
    rotated_atom_indices = struc.find_connected(
        bond_list_without_axis, root=atom_i
    )

    # Rotate
    coord[rotated_atom_indices] = struc.rotate_about_axis(
        coord[rotated_atom_indices], axis, angle, support
    )

    # replaced atom coords in larger molecule
    for i in rotated_atom_indices:
        a_i = residue[i]
        # --- find in larger molecule ---
        for j, atom_k in enumerate(mol):
            if atom_k.res_id == res_id and atom_k.atom_name == a_i.atom_name:
                #mol[j].translate(coord[i]-mol[j].coord)
                print(mol[j].coord)
                mol[j].coord += 10
                print(mol[j].coord)

    return mol


def plot(mol):

    colors = np.zeros((mol.array_length(), 3))
    colors[mol.element == "H"] = (0.8, 0.8, 0.8)  # gray
    colors[mol.element == "C"] = (0.0, 0.8, 0.0)  # green
    colors[mol.element == "N"] = (0.0, 0.0, 0.8)  # blue
    colors[mol.element == "O"] = (0.8, 0.0, 0.0)  # red

    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    graphics.plot_atoms(ax, mol, colors, line_width=3, zoom=1.5)
    fig.tight_layout()
    plt.show()


def plot(mol,mol2):


    fig = plt.figure(figsize=(16.0, 8.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.view_init(elev=0,azim=0)
    colors = np.zeros((mol.array_length(), 3))
    colors[mol.element == "H"] = (0.8, 0.8, 0.8)  # gray
    colors[mol.element == "C"] = (0.0, 0.8, 0.0)  # green
    colors[mol.element == "N"] = (0.0, 0.0, 0.8)  # blue
    colors[mol.element == "O"] = (0.8, 0.0, 0.0)  # red
    graphics.plot_atoms(ax, mol, colors, line_width=3, zoom=1.5)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.view_init(elev=0, azim=0)
    colors2 = np.zeros((mol2.array_length(), 3))
    colors2[mol2.element == "H"] = (0.8, 0.8, 0.8)  # gray
    colors2[mol2.element == "C"] = (0.0, 0.8, 0.0)  # green
    colors2[mol2.element == "N"] = (0.0, 0.0, 0.8)  # blue
    colors2[mol2.element == "O"] = (0.8, 0.0, 0.0)  # red
    graphics.plot_atoms(ax2, mol2, colors2, line_width=3, zoom=1.5)
    fig.tight_layout()
    plt.show()

# --- lets try and rotate/replace a residue in the peptide ---
sequence = seq.ProteinSequence("TIT")
mol = assemble_peptide(sequence)
mol2 = rotate_residue(mol, 2, 0, 45)
plot(mol, mol2)






