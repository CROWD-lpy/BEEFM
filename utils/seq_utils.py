import os
from rdkit import Chem, RDLogger
from tqdm import tqdm
import joblib
from collections import Counter
from utils.mol_operation import BondEditAction
from utils.edit_operation import check_edit_bond, check_edit_hydrogen_charge, atommapnum_minus_1, update_Hs_mapnum

RDLogger.DisableLog('rdApp.*')

def get_mols_from_edits(r_mol_dh, edits_bond_list, edits_hydrogen_list, bond_ptr, hydrogen_ptr):
    # print(edits_bond_list)
    # print(edits_hydrogen_list)
    # print(bond_ptr)
    # print(hydrogen_ptr)

    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in r_mol_dh.GetAtoms()}
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in r_mol_dh.GetAtoms()}

    mol_list = [r_mol_dh]; int_mol = r_mol_dh
    for mol_idx in range(len(bond_ptr) - 1):
        for edit_bond in edits_bond_list[bond_ptr[mol_idx]:bond_ptr[mol_idx+1]]:
            atom1, atom2, bond_order = edit_bond
            edit_exe = BondEditAction(atom_map1=atom1, atom_map2=atom2, bond_type=bond_order, action_vocab='test')
            int_mol = edit_exe.apply(Chem.Mol(int_mol))
        for edit_hydrogen in edits_hydrogen_list[hydrogen_ptr[mol_idx]:hydrogen_ptr[mol_idx+1]]:
            atom1, bond_order = edit_hydrogen
            bond_type = 0 if bond_order == -1 else 1
            hydrogens = []
            for atom in [bond.GetOtherAtomIdx(amap_to_idx[atom1])
                         for bond in r_mol_dh.GetAtoms()[amap_to_idx[atom1]].GetBonds()]:
                if r_mol_dh.GetAtoms()[atom].GetSymbol() == 'H': hydrogens.append(idx_to_amap[atom])
            if len(hydrogens) != 0:
                edit_exe = BondEditAction(atom_map1=atom1, atom_map2=hydrogens[0], bond_type=bond_type, action_vocab='test')
                int_mol = edit_exe.apply(Chem.Mol(int_mol))
            else:
                for update_idx, mol in enumerate(mol_list):
                    atom_new = Chem.Atom(1)
                    new_mol = Chem.RWMol(mol)
                    new_mol.AddAtom(atom_new)
                    new_mol.GetAtoms()[new_mol.GetNumAtoms()-1].SetAtomMapNum(new_mol.GetNumAtoms()-1)
                    mol_list[update_idx] = mol
                edit_exe = BondEditAction(atom_map1=atom1, atom_map2=new_mol.GetNumAtoms()-1, bond_type=bond_type, action_vocab='test')
                int_mol = edit_exe.apply(Chem.Mol(mol_list[-1]))
        mol_list.append(int_mol)
    return mol_list

def smiles2list(process_dir, args):
    data_path = f"./data/{args.dataset}/origin"

    data_path_ = os.path.join(data_path, "test.proc")
    with open(data_path_, 'r') as file: rxns = file.readlines()

    smiles = [rxn.split()[0] for rxn in rxns]
    edits  = [rxn.split()[1] for rxn in rxns]

    seq_smiles = []; idx = 0; current_smiles = []; current_edits = []
    while idx < len(smiles):
        if len(current_smiles) == 0:
            current_smiles.append(smiles[idx]); current_edits.append(edits[idx])
        elif current_smiles[-1].split('>>')[1] != smiles[idx].split('>>')[1]:
            seq_smiles.append((current_smiles, current_edits))
            current_smiles = []; current_edits = []
        else:
            current_smiles.append(smiles[idx]); current_edits.append(edits[idx])
        idx += 1

    rxns_data = []
    for rxn_idx, (smiles, edits_seq) in enumerate(tqdm(seq_smiles)):
        # print(edits_seq)
        bond_ptr = [0]; hydrogen_ptr = [0]
        edits_bond = []; edits_hydrogen = []
        for edits in edits_seq:
            edits_bond_temp = check_edit_bond(edits.split('/')[0]) if edits.split('/')[0] != '' else []
            edits_hydrogen_temp = check_edit_hydrogen_charge(edits.split('/')[1]) if edits.split('/')[1] != '' else []
            bond_ptr.append(bond_ptr[-1] + len(edits_bond_temp))
            hydrogen_ptr.append(hydrogen_ptr[-1] + len(edits_hydrogen_temp))
            edits_bond += edits_bond_temp
            if len(edits_hydrogen) != 0: edits_hydrogen += sorted(edits_hydrogen_temp, lambda x:x[1], reverse=True)

        r_smi, p_smi = smiles[0].split('>>')
        if args.dataset == 'coley':
            r_mol = Chem.MolFromSmiles(r_smi)
            if r_mol is None: continue
            r_mol = atommapnum_minus_1(r_mol)
            r_mol_dh = Chem.AddHs(r_mol)
            update_Hs_mapnum(r_mol_dh)
        else:
            r_mol = Chem.MolFromSmarts(r_smi)
            if r_mol is None: continue
            r_mol = atommapnum_minus_1(r_mol)
            r_mol_dh = r_mol

        mol_list = get_mols_from_edits(r_mol_dh, edits_bond, edits_hydrogen, bond_ptr, hydrogen_ptr)
        rxns_data.append((mol_list, len(smiles)))

    print(f'Save {len(rxns_data)} Reaction Sequences from {len(rxns)} Elementart Reactions...')
    joblib.dump(rxns_data, os.path.join(process_dir, 'seq_data.file'))


'''
100%|████████████████████████████████████| 42251/42251 [00:46<00:00, 905.49it/s]
Save 42078 Reaction Sequences from 90024 Elementart Reactions...
'''

def get_edits(r_mol_dh, edits_bond, edits_hydrogen):

    # if sum([edit_hydrogen[1] for edit_hydrogen in edits_hydrogen]) > 0:
    #     for edit_hydrogen in edits_hydrogen:
    #         assert edit_hydrogen[1] == 1
    #         atom_new = Chem.Atom(1)
    #         r_mol_dh = Chem.RWMol(r_mol_dh)
    #         r_mol_dh.AddAtom(atom_new)
    #         r_mol_dh.GetAtoms()[r_mol_dh.GetNumAtoms() - 1].SetAtomMapNum(r_mol_dh.GetNumAtoms() - 1)

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
        # edits_hydrogen = sorted(edits_hydrogen, key=lambda x: x[1])
        hydrogens = []
        for edit_hydrogen in edits_hydrogen:
            atom1, bond_order = edit_hydrogen
            if bond_order == -1:
                if len(hydrogens) > 0: hydrogens = []
                for atom in [bond.GetOtherAtomIdx(amap_to_idx[atom1])
                             for bond in r_mol_dh.GetAtoms()[amap_to_idx[atom1]].GetBonds()]:
                    if r_mol_dh.GetAtoms()[atom].GetSymbol() == 'H': hydrogens.append(idx_to_amap[atom])
                # assert len(hydrogens) > 0
                if len(hydrogens) == 0: return None, []
                edits_rxn['B'].append((atom1, sorted(hydrogens)[0], 0.0))
            else:
                if len(hydrogens) == 0: return None, []
                atom1, bond_order = edit_hydrogen
                edits_rxn['F'].append((atom1, sorted(hydrogens)[0], 1.0))
                hydrogen = []
    else:
        return None, []
        for hydrogen_idx, edit_hydrogen in enumerate(edits_hydrogen):
            atom1, _ = edit_hydrogen
            edits_rxn['F'].append((atom1, r_mol_dh.GetNumAtoms() - 1 - hydrogen_idx, 1.0))

    return r_mol_dh, edits_rxn

def all_mol_check(smiles):
    mols = []
    for smile_idx, smile in enumerate(smiles):
        r_smile, p_smile = smile.split('>>')
        r_mol, p_mol = Chem.MolFromSmiles(r_smile), Chem.MolFromSmiles(p_smile)
        if smile_idx == 0: mols.append(r_mol); num_atoms = r_mol.GetNumAtoms()
        if r_mol.GetNumAtoms() == p_mol.GetNumAtoms() == num_atoms:
            mols.append(p_mol)
        else:
            return False
        if r_mol is None or p_mol is None: return False
    return mols


def get_seq_from_proc(process_dir, args):
    data_path = f"./data/{args['dataset']}/origin"
    data_path_ = os.path.join(data_path, "test.proc")

    with open(data_path_, 'r') as file: lines = file.readlines()
    smiles = [line.split()[0] for line in lines]
    edits = [line.split()[1] for line in lines]

    idx = 0; rxns = []; current_edits = []; current_smile = []
    while idx < len(smiles):
        print(idx)
        if edits[idx] == '//':
            if len(current_smile) == 0: current_edits = []; current_smile = []; idx += 1; continue
            mols = all_mol_check(current_smile)
            if not mols: current_edits = []; current_smile = []; idx += 1; continue
            mols_dh = []
            for mol in mols:
                mol = atommapnum_minus_1(mol)
                mol_dh = Chem.AddHs(mol)
                update_Hs_mapnum(mol_dh)
                mols_dh.append(mol_dh)
            if len(mols_dh) > 10: current_edits = []; current_smile = []; idx += 1; continue

            edits_bonds = ';'.join([x.split('/')[0] for x in current_edits])
            edits_bonds = check_edit_bond(edits_bonds) if edits_bonds != '' else []
            edits_hydrogen = []
            for x in current_edits:
                if x.split('/')[1] != '':
                    edit_hydrogen = check_edit_hydrogen_charge(x.split('/')[1])
                    edit_hydrogen = sorted(edit_hydrogen, key=lambda x:x[1], reverse=False)
                    edits_hydrogen += edit_hydrogen


            r_mol_dh, edits_list = get_edits(r_mol_dh, edits_bonds, edits_hydrogen)
            if r_mol_dh is not None:
                edits_list = edits_list['B'] + edits_list['C'] + edits_list['F']
                int_mol = r_mol_dh
                for edit in edits_list:
                    if edit[2] != -1:
                        edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2],
                                                      action_vocab='test')
                        int_mol = edit_exe.apply(Chem.Mol(int_mol))
                rxns.append((r_mol_dh, int_mol, len(current_edits) + 1, edits_list))
            current_edits = []; current_smile = []
        elif current_smile is None:
            current_smile.append(smiles[idx]); current_edits.append(edits[idx])
        else:
            current_edits.append(edits[idx])
        idx += 1
    print(len(rxns))
    print(rxns[0])
    joblib.dump(rxns, process_dir)

