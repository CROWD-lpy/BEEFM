from rdkit import Chem
import random

def check_edit_bond(edits_str):
    ans = []
    for edit_str in edits_str.split(';'):
        if edit_str == '': continue
        ans.append(
            (
                int(edit_str.split('-')[0]) - 1,
                int(edit_str.split('-')[1]) - 1,
                float(edit_str.split('-')[2])
            )
        )
    return ans

def check_edit_hydrogen_charge(edits_str):
    ans = []
    for edit_str in edits_str.split(';'):
        if edit_str == '': continue
        ans.append(
            (
                int(edit_str.split(':')[0]) - 1,
                int(edit_str.split(':')[1])
            )
        )
    return ans

def atommapnum_minus_1(mol):
    # map_to_idx = {}
    # idx_to_map = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0: continue
        atom.SetAtomMapNum(atom.GetAtomMapNum() - 1)
        # map_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()
        # idx_to_map[atom.GetIdx()] = atom.GetAtomMapNum()
    return mol


def update_Hs_mapnum(mol):
    atom_num = mol.GetNumAtoms(); mapnum = max([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    atoms = mol.GetAtoms(); current_mapnum = mapnum + 1
    sorted_atoms = sorted(atoms, key=lambda atom: atom.GetAtomMapNum())
    for atom in sorted_atoms:
        if atom.GetSymbol() == 'H' and atom.GetAtomMapNum() == 0: continue
        bonds = atom.GetBonds()
        connected_atoms = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in bonds]
        for atom_idx in connected_atoms:
            atom_ = mol.GetAtomWithIdx(atom_idx)
            if atom_.GetSymbol() == 'H' and atom_.GetAtomMapNum() == 0:
                atom_.SetProp("molAtomMapNumber", str(current_mapnum))
                current_mapnum += 1


def get_edits(r_mol_dh, edits_bond, edits_hydrogen):

    if sum([edit_hydrogen[1] for edit_hydrogen in edits_hydrogen]) > 0:
        for edit_hydrogen in edits_hydrogen:
            assert edit_hydrogen[1] == 1
            atom_new = Chem.Atom(1)
            r_mol_dh = Chem.RWMol(r_mol_dh)
            r_mol_dh.AddAtom(atom_new)
            r_mol_dh.GetAtoms()[r_mol_dh.GetNumAtoms() - 1].SetAtomMapNum(r_mol_dh.GetNumAtoms() - 1)

    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in r_mol_dh.GetAtoms()}
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in r_mol_dh.GetAtoms()}

    edits_rxn = {'B': [], 'C': [], 'F': []} # break/change/formation
    for edit_bond in edits_bond:
        atom1, atom2, bond_order = edit_bond
        bond_type_prv = r_mol_dh.GetBondBetweenAtoms(amap_to_idx[atom1], amap_to_idx[atom2])
        bond_order_prv = 0 if bond_type_prv is None else int(bond_type_prv.GetBondTypeAsDouble())
        # if int(bond_order_prv) == bond_order_prv: bond_order_prv = int(bond_order_prv)
        if bond_order_prv != 0 and bond_order == 0:
            edits_rxn['B'].append(edit_bond)
        elif bond_order_prv != 0 and bond_order != bond_order_prv:
            edits_rxn['C'].append(edit_bond)
        elif bond_order_prv == 0 and bond_order > 0:
            edits_rxn['F'].append(edit_bond)

    if sum([edit_hydrogen[1] for edit_hydrogen in edits_hydrogen]) <= 0:
        edits_hydrogen = sorted(edits_hydrogen, key=lambda x: x[1])
        for edit_hydrogen in edits_hydrogen:
            atom1, bond_order = edit_hydrogen
            if bond_order == -1:
                hydrogens = []
                for atom in [bond.GetOtherAtomIdx(amap_to_idx[atom1])
                             for bond in r_mol_dh.GetAtoms()[amap_to_idx[atom1]].GetBonds()]:
                    if r_mol_dh.GetAtoms()[atom].GetSymbol() == 'H': hydrogens.append(idx_to_amap[atom])
                assert len(hydrogens) > 0
                edits_rxn['B'].append((atom1, sorted(hydrogens)[0], 0.0))
            else:
                atom1, bond_order = edits_hydrogen[1]
                assert bond_order == 1
                edits_rxn['F'].append((atom1, sorted(hydrogens)[0], 1.0))
    else:
        for hydrogen_idx, edit_hydrogen in enumerate(edits_hydrogen):
            atom1, _ = edit_hydrogen
            edits_rxn['F'].append((atom1, r_mol_dh.GetNumAtoms() - 1 - hydrogen_idx, 1.0))

    return r_mol_dh, edits_rxn


def shuffle_edits(r_mol_dh, edits_rxn, edit_order, seed=None):
    if seed is not None: random.seed(seed)
    shuffle_list = list(range(r_mol_dh.GetNumAtoms()))
    random.shuffle(shuffle_list)
    for atom in r_mol_dh.GetAtoms():
        atom.SetAtomMapNum(shuffle_list[atom.GetAtomMapNum()])
    random.shuffle(edits_rxn['B'])
    random.shuffle(edits_rxn['F'])
    random.shuffle(edits_rxn['C'])

    edits_rxn_list = []
    for type in edit_order:
        for edit in edits_rxn[type]:
            edits_rxn_list.append((shuffle_list[edit[0]], shuffle_list[edit[1]], edit[2]))

    return r_mol_dh, edits_rxn_list

def check_aro(edits_rxn):
    edits_list = []
    for key in edits_rxn.keys():
        edits_list += edits_rxn[key]
    for edit in edits_list:
        if edit[2] == 1.5:
            return True
    return False

def update_statistic(edits_list, edits_type, edits_length):
    if len(edits_list) not in edits_length:
        edits_length[len(edits_list)] = 1
    else:
        edits_length[len(edits_list)] += 1
    for edit in edits_list:
        if edit[2] not in edits_type:
            edits_type[edit[2]] = 1
        else:
            edits_type[edit[2]] += 1
    return edits_type, edits_length