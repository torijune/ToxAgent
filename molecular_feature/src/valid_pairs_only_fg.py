"""
Validation-by-Reconstruction for only_fg pairs.

This script validates that pairs differ ONLY in functional groups by:
1. Extracting FG information using AccFG
2. Removing unique toxic FGs from toxic molecule
3. Adding unique nontoxic FGs from nontoxic molecule
4. Comparing reconstructed canonical SMILES with original nontoxic canonical SMILES

Based on FGBench validation-by-reconstruction approach.
"""
import sys
from pathlib import Path
import json
import ast
from typing import Dict, List, Optional, Tuple, Set

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Try different AccFG import paths
ACCFG_AVAILABLE = False
AccFG = None
accfg_functions_available = False

try:
    # Try AccFG_private path (FGBench style)
    import os
    PROJECT_DIR = os.getcwd()
    ACCFG_DIR = os.path.join(PROJECT_DIR, 'AccFG_private')
    if os.path.exists(ACCFG_DIR):
        sys.path.append(ACCFG_DIR)
    
    from accfg import (
        AccFG, remove_fg_list_from_mol, get_RascalMCES,
        get_outer_bond_from_fg_list, set_atom_idx
    )
    ACCFG_AVAILABLE = True
    accfg_functions_available = True
except ImportError:
    try:
        from AccFG.accfg import AccFG
        ACCFG_AVAILABLE = True
    except ImportError:
        try:
            from AccFG import AccFG
            ACCFG_AVAILABLE = True
        except ImportError:
            try:
                from accfg import AccFG
                ACCFG_AVAILABLE = True
            except ImportError:
                print("⚠️ Warning: AccFG library not found. Validation will be limited.")
                print("   Please install AccFG for full validation functionality.")

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


def parse_fg_full(fg_full_str: str) -> Dict[str, List[List[int]]]:
    """Parse fg_full JSON string to dictionary.
    
    Args:
        fg_full_str: JSON string representation of fg_full
        
    Returns:
        Dictionary mapping FG names to lists of atom index lists
    """
    if pd.isna(fg_full_str) or not fg_full_str:
        return {}
    try:
        if isinstance(fg_full_str, dict):
            return fg_full_str
        if isinstance(fg_full_str, str):
            try:
                return json.loads(fg_full_str)
            except:
                return ast.literal_eval(fg_full_str)
    except Exception as e:
        print(f"Warning: Failed to parse fg_full: {e}")
    return {}


def parse_unique_fg(unique_fg_str: str) -> List[Dict]:
    """Parse unique_fg JSON string to list of dictionaries.
    
    Args:
        unique_fg_str: JSON string representation of unique_fg
        
    Returns:
        List of unique FG dictionaries
    """
    if pd.isna(unique_fg_str) or not unique_fg_str:
        return []
    try:
        if isinstance(unique_fg_str, list):
            return unique_fg_str
        if isinstance(unique_fg_str, str):
            try:
                return json.loads(unique_fg_str)
            except:
                return ast.literal_eval(unique_fg_str)
    except Exception as e:
        print(f"Warning: Failed to parse unique_fg: {e}")
    return []


def merge_diff_tuple(diff_tuple_list: List) -> List:
    """Merge diff tuple list (FGBench style).
    
    This function follows FGBench's exact implementation:
    - Handles nested lists: [[tuple1], [tuple2]] -> [tuple1, tuple2]
    - Preserves flat lists: [tuple1, tuple2] -> [tuple1, tuple2]
    
    Args:
        diff_tuple_list: List of diff tuples (can be nested or flat)
        
    Returns:
        Merged list of diff tuples (flat list of tuples)
    """
    merged_diff = []
    for diff_tuple in diff_tuple_list:
        # Check if it's a nested list (list of tuples) or a single tuple
        if isinstance(diff_tuple, list):
            # If it's a list, check if it contains tuples
            if diff_tuple and isinstance(diff_tuple[0], tuple):
                # Nested list: [[tuple1], [tuple2]] -> add each tuple
                merged_diff.extend(diff_tuple)
            else:
                # Flat list of tuples: [tuple1, tuple2] -> add all
                merged_diff.extend(diff_tuple)
        elif isinstance(diff_tuple, tuple):
            # Single tuple: add it directly
            merged_diff.append(diff_tuple)
        else:
            # Fallback: try to add as-is (shouldn't happen)
            merged_diff.append(diff_tuple)
    return merged_diff


def get_unique_fg_diff_tuples(
    unique_fg_list: List[Dict],
    toxic_fg_full: Dict[str, List[List[int]]],
    nontoxic_fg_full: Dict[str, List[List[int]]],
    reason: str
) -> List[Tuple]:
    """Get FG diff tuples for unique FGs with given reason (FGBench style).
    
    Args:
        unique_fg_list: List of unique FG dictionaries
        toxic_fg_full: Toxic FG full dictionary with atom indices
        nontoxic_fg_full: Nontoxic FG full dictionary with atom indices
        reason: 'name_only_in_toxic', 'name_only_in_nontoxic', or 'atom_index_diff'
        
    Returns:
        List of (fg_name, fg_smiles, fg_atoms) tuples
    """
    diff_tuples = []
    for fg_info in unique_fg_list:
        if not isinstance(fg_info, dict):
            continue
        if fg_info.get('reason') != reason:
            continue
        fg_name = fg_info.get('fg_name')
        if not fg_name:
            continue
        
        # Determine which fg_full to use based on reason
        if reason == 'name_only_in_toxic':
            # Use toxic_fg_full
            fg_full_to_use = toxic_fg_full
            atom_indices_key = 'toxic_atom_indices'
        elif reason == 'name_only_in_nontoxic':
            # Use nontoxic_fg_full
            fg_full_to_use = nontoxic_fg_full
            atom_indices_key = 'nontoxic_atom_indices'
        else:  # atom_index_diff
            # Use toxic_fg_full (or both, but we'll use toxic for now)
            fg_full_to_use = toxic_fg_full
            atom_indices_key = 'toxic_atom_indices'
        
        # Try to get atom indices from fg_info first (more reliable)
        atom_indices = fg_info.get(atom_indices_key, [])
        if atom_indices:
            # Use atom indices from unique_fg_info
            for atom_item in atom_indices:
                if isinstance(atom_item, (list, tuple)):
                    # atom_item is one FG instance: a list/tuple of atomNote indices
                    # AccFG expects fg_atoms as List[List[int]] (list of atom-lists)
                    group = list(atom_item)
                    diff_tuples.append((fg_name, fg_name, [group]))
                elif isinstance(atom_item, int):
                    # Single atom index -> one group with one atom
                    diff_tuples.append((fg_name, fg_name, [[atom_item]]))
        elif fg_name in fg_full_to_use:
            # Fallback: use fg_full
            for atom_item in fg_full_to_use[fg_name]:
                if isinstance(atom_item, (list, tuple)):
                    # atom_item is one FG instance: list/tuple of atom indices
                    group = list(atom_item)
                    diff_tuples.append((fg_name, fg_name, [group]))
                elif isinstance(atom_item, int):
                    diff_tuples.append((fg_name, fg_name, [[atom_item]]))
    
    return diff_tuples


def remove_atoms_from_mol(mol: Chem.Mol, atoms_to_remove: Set[int]) -> Optional[Chem.Mol]:
    """Remove specified atoms from molecule.
    
    Args:
        mol: RDKit molecule
        atoms_to_remove: Set of atom indices to remove
        
    Returns:
        New molecule with atoms removed, or None if removal fails
    """
    if not mol or not atoms_to_remove:
        return mol
    
    try:
        # Create editable molecule
        emol = Chem.EditableMol(mol)
        
        # Sort atoms in descending order to avoid index shifting issues
        atoms_sorted = sorted(atoms_to_remove, reverse=True)
        
        # Remove atoms
        for atom_idx in atoms_sorted:
            if atom_idx < mol.GetNumAtoms():
                emol.RemoveAtom(atom_idx)
        
        # Get new molecule
        new_mol = emol.GetMol()
        
        # Sanitize
        try:
            Chem.SanitizeMol(new_mol)
        except:
            # If sanitization fails, try to fix
            try:
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
            except:
                return None
        
        return new_mol
    except Exception as e:
        print(f"Error removing atoms: {e}")
        return None


def extract_fg_subgraph(mol: Chem.Mol, fg_atoms: Set[int]) -> Optional[Chem.Mol]:
    """Extract subgraph containing specified atoms.
    
    Args:
        mol: RDKit molecule
        fg_atoms: Set of atom indices in the FG
        
    Returns:
        Subgraph molecule, or None if extraction fails
    """
    if not mol or not fg_atoms:
        return None
    
    try:
        # Get all atoms and bonds in the subgraph
        atoms_to_keep = set(fg_atoms)
        
        # Add neighboring atoms if needed (for attachment points)
        # For now, just extract the exact atoms
        emol = Chem.EditableMol(mol)
        
        # Find atoms to remove (all atoms NOT in fg_atoms)
        atoms_to_remove = set(range(mol.GetNumAtoms())) - atoms_to_keep
        
        # Remove atoms not in FG
        for atom_idx in sorted(atoms_to_remove, reverse=True):
            emol.RemoveAtom(atom_idx)
        
        new_mol = emol.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except:
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        
        return new_mol
    except Exception:
        return None


def attach_fg_to_mol(
    base_mol: Chem.Mol,
    fg_mol: Chem.Mol,
    attachment_atom_base: Optional[int] = None,
    attachment_atom_fg: Optional[int] = None
) -> Optional[Chem.Mol]:
    """Attach FG molecule to base molecule.
    
    This is a simplified version. In practice, finding the correct
    attachment points is complex.
    
    Args:
        base_mol: Base molecule (after removing toxic FGs)
        fg_mol: FG molecule to attach
        attachment_atom_base: Atom index in base_mol to attach to
        attachment_atom_fg: Atom index in fg_mol to attach from
        
    Returns:
        Combined molecule, or None if attachment fails
    """
    if not base_mol or not fg_mol:
        return None
    
    try:
        # Simple approach: combine SMILES and let RDKit handle it
        # This is not perfect but works for many cases
        base_smi = Chem.MolToSmiles(base_mol)
        fg_smi = Chem.MolToSmiles(fg_mol)
        
        # Try to combine (this is a heuristic)
        # If base_mol has attachment points, we need to handle them
        combined_smi = f"{base_smi}.{fg_smi}"
        combined_mol = Chem.MolFromSmiles(combined_smi)
        
        if combined_mol:
            return combined_mol
        
        # If that fails, try direct concatenation (for terminal FGs)
        # This is a simplified approach
        return None
    except Exception:
        return None


def exam_comparison(
    target_smiles: str,
    ref_smiles: str,
    target_diff: List,
    ref_diff: List
) -> bool:
    """Examine if two molecules have the same scaffold after removing FGs (FGBench style).
    
    This function follows FGBench's exact implementation:
    1. Parse molecules from SMILES
    2. Set atom indices using set_atom_idx with 'atomNote'
    3. Merge diff tuples using merge_diff_tuple
    4. Remove FGs using remove_fg_list_from_mol
    5. Compare canonical SMILES (non-isomeric)
    
    Args:
        target_smiles: Target molecule SMILES
        ref_smiles: Reference molecule SMILES
        target_diff: List of FG diff tuples for target molecule (can be nested)
        ref_diff: List of FG diff tuples for reference molecule (can be nested)
        
    Returns:
        True if scaffolds match after removing FGs, False otherwise
    """
    try:
        # Parse molecules
        target_mol = Chem.MolFromSmiles(target_smiles)
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        
        if not target_mol or not ref_mol:
            return False
        
        # Merge diff tuples (handles nested structures)
        target_fg_diff = merge_diff_tuple(target_diff)
        ref_fg_diff = merge_diff_tuple(ref_diff)
        
        # Optimization: If one side has no unique FGs, compare directly
        # Case 1: ref_diff is empty (M_nt has no unique FGs)
        if not ref_fg_diff:
            # Remove FGs from target and compare with ref directly
            if accfg_functions_available:
                target_mol = set_atom_idx(target_mol, 'atomNote')
                ref_mol = set_atom_idx(ref_mol, 'atomNote')
                target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
                if not target_remain_mol:
                    return False
                Chem.SanitizeMol(target_remain_mol)
                target_remain_smi = Chem.MolToSmiles(target_remain_mol, isomericSmiles=False)
                ref_smi = Chem.MolToSmiles(ref_mol, isomericSmiles=False)
                return target_remain_smi == ref_smi
            else:
                # Fallback: simple atom removal
                target_atoms = set()
                for diff_tuple in target_fg_diff:
                    if len(diff_tuple) >= 3:
                        atoms = diff_tuple[2]
                        if isinstance(atoms, (list, tuple)):
                            target_atoms.update(atoms)
                        elif isinstance(atoms, int):
                            target_atoms.add(atoms)
                target_remain_mol = remove_atoms_from_mol(target_mol, target_atoms)
                if not target_remain_mol:
                    return False
                Chem.SanitizeMol(target_remain_mol)
                target_remain_smi = Chem.MolToSmiles(target_remain_mol, isomericSmiles=False)
                ref_smi = Chem.MolToSmiles(ref_mol, isomericSmiles=False)
                return target_remain_smi == ref_smi
        
        # Case 2: target_diff is empty (M_t has no unique FGs)
        elif not target_fg_diff:
            # Remove FGs from ref and compare with target directly
            if accfg_functions_available:
                target_mol = set_atom_idx(target_mol, 'atomNote')
                ref_mol = set_atom_idx(ref_mol, 'atomNote')
                ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
                if not ref_remain_mol:
                    return False
                Chem.SanitizeMol(ref_remain_mol)
                ref_remain_smi = Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)
                target_smi = Chem.MolToSmiles(target_mol, isomericSmiles=False)
                return target_smi == ref_remain_smi
            else:
                # Fallback: simple atom removal
                ref_atoms = set()
                for diff_tuple in ref_fg_diff:
                    if len(diff_tuple) >= 3:
                        atoms = diff_tuple[2]
                        if isinstance(atoms, (list, tuple)):
                            ref_atoms.update(atoms)
                        elif isinstance(atoms, int):
                            ref_atoms.add(atoms)
                ref_remain_mol = remove_atoms_from_mol(ref_mol, ref_atoms)
                if not ref_remain_mol:
                    return False
                Chem.SanitizeMol(ref_remain_mol)
                ref_remain_smi = Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)
                target_smi = Chem.MolToSmiles(target_mol, isomericSmiles=False)
                return target_smi == ref_remain_smi
        
        # Case 3: Both sides have unique FGs (original FGBench logic)
        else:
            # Use AccFG functions if available (FGBench style)
            if accfg_functions_available:
                # Set atom indices using atomNote (FGBench style)
                target_mol = set_atom_idx(target_mol, 'atomNote')
                ref_mol = set_atom_idx(ref_mol, 'atomNote')
                
                # Remove FGs from both molecules using AccFG
                target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
                ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
                
                if not target_remain_mol or not ref_remain_mol:
                    return False
                
                # Sanitize
                Chem.SanitizeMol(target_remain_mol)
                Chem.SanitizeMol(ref_remain_mol)
                
                # Compare canonical SMILES (non-isomeric)
                target_remain_smi = Chem.MolToSmiles(target_remain_mol, isomericSmiles=False)
                ref_remain_smi = Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)
                
                return target_remain_smi == ref_remain_smi
        
            else:
                # Fallback: use simple atom removal (less accurate but works without AccFG)
                # Collect atoms to remove
                target_atoms = set()
                for diff_tuple in target_fg_diff:
                    if len(diff_tuple) >= 3:
                        atoms = diff_tuple[2]  # (fg_name, fg_smiles, fg_atoms)
                        if isinstance(atoms, (list, tuple)):
                            target_atoms.update(atoms)
                        elif isinstance(atoms, int):
                            target_atoms.add(atoms)
                
                ref_atoms = set()
                for diff_tuple in ref_fg_diff:
                    if len(diff_tuple) >= 3:
                        atoms = diff_tuple[2]  # (fg_name, fg_smiles, fg_atoms)
                        if isinstance(atoms, (list, tuple)):
                            ref_atoms.update(atoms)
                        elif isinstance(atoms, int):
                            ref_atoms.add(atoms)
                
                # Remove atoms
                target_remain_mol = remove_atoms_from_mol(target_mol, target_atoms)
                ref_remain_mol = remove_atoms_from_mol(ref_mol, ref_atoms)
                
                if not target_remain_mol or not ref_remain_mol:
                    return False
                
                # Sanitize
                Chem.SanitizeMol(target_remain_mol)
                Chem.SanitizeMol(ref_remain_mol)
                
                # Compare canonical SMILES (non-isomeric)
                target_remain_smi = Chem.MolToSmiles(target_remain_mol, isomericSmiles=False)
                ref_remain_smi = Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)
                
                return target_remain_smi == ref_remain_smi
            
    except Exception as e:
        import traceback
        print(f"Error in exam_comparison: {e}")
        traceback.print_exc()
        return False


def validate_pair_by_reconstruction(
    toxic_smiles: str,
    nontoxic_smiles: str,
    toxic_canonical: str,
    nontoxic_canonical: str,
    toxic_fg_full: Dict[str, List[List[int]]],
    nontoxic_fg_full: Dict[str, List[List[int]]],
    unique_fg_list: List[Dict],
    afg: AccFG
) -> Tuple[bool, str, Optional[str]]:
    """Validate pair by reconstruction approach (FGBench style).
    
    Uses FGBench's exam_comparison function to check if scaffolds match
    after removing unique FGs from both molecules.
    
    Args:
        toxic_smiles: Toxic SMILES
        nontoxic_smiles: Nontoxic SMILES
        toxic_canonical: Toxic canonical SMILES
        nontoxic_canonical: Nontoxic canonical SMILES
        toxic_fg_full: Toxic FG full dictionary
        nontoxic_fg_full: Nontoxic FG full dictionary
        unique_fg_list: List of unique FG dictionaries
        afg: AccFG instance
        
    Returns:
        Tuple of (is_valid, error_message, reconstructed_smiles)
    """
    try:
        # Get unique FG diff tuples (FGBench style)
        # For toxic (target): name_only_in_toxic + atom_index_diff (toxic side)
        toxic_diff = get_unique_fg_diff_tuples(
            unique_fg_list, toxic_fg_full, nontoxic_fg_full, 'name_only_in_toxic'
        )
        
        # For nontoxic (ref): name_only_in_nontoxic + atom_index_diff (nontoxic side)
        nontoxic_diff = get_unique_fg_diff_tuples(
            unique_fg_list, toxic_fg_full, nontoxic_fg_full, 'name_only_in_nontoxic'
        )
        
        # Handle atom_index_diff: need to get both toxic and nontoxic atom indices
        atom_diff_toxic = []
        atom_diff_nontoxic = []
        for fg_info in unique_fg_list:
            if not isinstance(fg_info, dict):
                continue
            if fg_info.get('reason') != 'atom_index_diff':
                continue
            fg_name = fg_info.get('fg_name')
            if not fg_name:
                continue
            
            # Get toxic atom indices
            toxic_atom_indices = fg_info.get('toxic_atom_indices', [])
            for atom_item in toxic_atom_indices:
                if isinstance(atom_item, (list, tuple)):
                    # AccFG expects List[List[int]] (list of atom-lists)
                    group = list(atom_item)
                    atom_diff_toxic.append((fg_name, fg_name, [group]))
                elif isinstance(atom_item, int):
                    atom_diff_toxic.append((fg_name, fg_name, [[atom_item]]))
            
            # Get nontoxic atom indices
            nontoxic_atom_indices = fg_info.get('nontoxic_atom_indices', [])
            for atom_item in nontoxic_atom_indices:
                if isinstance(atom_item, (list, tuple)):
                    group = list(atom_item)
                    atom_diff_nontoxic.append((fg_name, fg_name, [group]))
                elif isinstance(atom_item, int):
                    atom_diff_nontoxic.append((fg_name, fg_name, [[atom_item]]))
        
        # Combine: target_diff = toxic unique + atom_diff (toxic side)
        target_diff = toxic_diff + atom_diff_toxic
        
        # ref_diff = nontoxic unique + atom_diff (nontoxic side)
        ref_diff = nontoxic_diff + atom_diff_nontoxic
        
        # If no unique FGs, skip (shouldn't happen for only_fg pairs)
        if not target_diff and not ref_diff:
            return False, "No unique FGs found", None
        
        # Use FGBench's exam_comparison function
        # toxic = target, nontoxic = ref
        # exam_comparison handles AccFG availability internally
        is_valid = exam_comparison(
            toxic_canonical,
            nontoxic_canonical,
            target_diff,
            ref_diff
        )
        
        if is_valid:
            return True, "Valid: Scaffolds match after FG removal", None
        else:
            return False, "Invalid: Scaffolds do not match after FG removal", None
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}", None


def validate_pairs(
    input_csv: Path,
    output_csv: Path,
    limit: Optional[int] = None
) -> None:
    """Validate only_fg pairs using reconstruction approach.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        limit: Optional limit on number of pairs to process
    """
    print(f"Loading pairs from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Filter to only_fg
    df_fg = df[df['diff_type'] == 'only_fg'].copy()
    print(f"only_fg rows: {len(df_fg)}")
    
    if len(df_fg) == 0:
        print("No only_fg pairs found. Exiting.")
        return
    
    # Initialize AccFG
    if not ACCFG_AVAILABLE:
        print("❌ Error: AccFG is required for validation. Please install AccFG.")
        return
    
    print("Initializing AccFG...")
    if accfg_functions_available:
        # Use lite mode for faster processing (FGBench style)
        afg = AccFG(lite=True, print_load_info=False)
    else:
        afg = AccFG(lite=False, print_load_info=False)
    
    # Process pairs
    valid_rows = []
    invalid_rows = []
    
    df_to_process = df_fg.head(limit) if limit else df_fg
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Validating pairs"):
        try:
            # Extract data
            toxic_smiles = str(row.get('toxic_smiles', '')).strip()
            nontoxic_smiles = str(row.get('nontoxic_smiles', '')).strip()
            toxic_canonical = str(row.get('toxic_canonical_smiles', '')).strip()
            nontoxic_canonical = str(row.get('nontoxic_canonical_smiles', '')).strip()
            
            if not toxic_smiles or not nontoxic_smiles or pd.isna(toxic_smiles) or pd.isna(nontoxic_smiles):
                invalid_rows.append({
                    'row_index': idx,
                    'reason': 'Missing SMILES',
                    'original_row': row.to_dict()
                })
                continue
            
            # Parse FG data
            toxic_fg_full = parse_fg_full(row.get('toxic_fg_full', '{}'))
            nontoxic_fg_full = parse_fg_full(row.get('nontoxic_fg_full', '{}'))
            unique_fg_list = parse_unique_fg(row.get('unique_fg', '[]'))
            
            # Validate
            is_valid, error_msg, reconstructed = validate_pair_by_reconstruction(
                toxic_smiles,
                nontoxic_smiles,
                toxic_canonical,
                nontoxic_canonical,
                toxic_fg_full,
                nontoxic_fg_full,
                unique_fg_list,
                afg
            )
            
            if is_valid:
                # Add validation status to row
                row_dict = row.to_dict()
                row_dict['validation_status'] = 'valid'
                row_dict['validation_message'] = error_msg
                valid_rows.append(row_dict)
            else:
                invalid_rows.append({
                    'row_index': idx,
                    'reason': error_msg,
                    'original_row': row.to_dict()
                })
                
        except Exception as e:
            invalid_rows.append({
                'row_index': idx,
                'reason': f'Exception: {str(e)}',
                'original_row': row.to_dict()
            })
    
    # Create output DataFrame
    if valid_rows:
        df_valid = pd.DataFrame(valid_rows)
        df_valid.to_csv(output_csv, index=False)
        print(f"\n✅ Validated {len(valid_rows)} pairs")
        print(f"✅ Saved to: {output_csv}")
    else:
        print(f"\n⚠️ No valid pairs found")
    
    if invalid_rows:
        print(f"❌ Invalid pairs: {len(invalid_rows)}")
        # Optionally save invalid pairs for debugging
        invalid_csv = output_csv.parent / f"{output_csv.stem}_invalid.csv"
        df_invalid = pd.DataFrame(invalid_rows)
        df_invalid.to_csv(invalid_csv, index=False)
        print(f"   Invalid pairs saved to: {invalid_csv}")


def main():
    """Main function."""
    input_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot.csv'
    output_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot_validated.csv'
    
    validate_pairs(input_csv, output_csv)


if __name__ == "__main__":
    main()
