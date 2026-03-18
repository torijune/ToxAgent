"""
Visualize sample valid and invalid pairs.

For each sample, shows:
1. Toxic molecule (original)
2. Nontoxic molecule (original)
3. Toxic molecule after removing unique FGs
4. Nontoxic molecule after removing unique FGs
"""
import sys
from pathlib import Path
import pandas as pd
import ast
from typing import Dict, List, Optional, Tuple, Set

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

# Import validation functions
from molecular_feature.src.valid_pairs_only_fg import (
    parse_fg_full,
    get_unique_fg_diff_tuples,
    remove_atoms_from_mol,
    merge_diff_tuple,
    accfg_functions_available
)

# Try to import AccFG functions
try:
    import sys
    from pathlib import Path
    accfg_path = Path(__file__).parent.parent.parent / "AccFG_private"
    if accfg_path.exists():
        sys.path.insert(0, str(accfg_path))
    from accfg.compare import remove_fg_list_from_mol, set_atom_idx
    ACCFG_AVAILABLE = True
except ImportError:
    ACCFG_AVAILABLE = False
    remove_fg_list_from_mol = None
    set_atom_idx = None


def get_removed_smiles(
    smiles: str,
    fg_diff: List[Tuple],
    fg_full: Dict[str, List[List[int]]]
) -> Optional[str]:
    """Get SMILES after removing FGs (using same method as exam_comparison).
    
    This function uses the same approach as exam_comparison:
    1. Set atom indices using set_atom_idx with 'atomNote' (if AccFG available)
    2. Remove FGs using remove_fg_list_from_mol (if AccFG available)
    3. Fallback to direct atom removal if AccFG not available
    
    Args:
        smiles: Original SMILES
        fg_diff: List of (fg_name, fg_smiles, fg_atoms) tuples
        fg_full: FG full dictionary for atom indices (not used if AccFG available)
        
    Returns:
        SMILES after removal, or None if fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Merge diff tuples (handles nested structures)
    merged_fg_diff = merge_diff_tuple(fg_diff)
    
    if not merged_fg_diff:
        return smiles
    
    # Try AccFG method first (same as exam_comparison)
    if ACCFG_AVAILABLE and remove_fg_list_from_mol and set_atom_idx:
        try:
            # Set atom indices using atomNote (FGBench style)
            mol_with_idx = set_atom_idx(mol, 'atomNote')
            remain_mol = remove_fg_list_from_mol(mol_with_idx, merged_fg_diff)
            if remain_mol:
                Chem.SanitizeMol(remain_mol)
                return Chem.MolToSmiles(remain_mol, isomericSmiles=False)
        except Exception as e:
            print(f"Warning: AccFG method failed, using fallback: {e}")
    
    # Fallback: direct atom removal
    atoms_to_remove = set()
    for diff_tuple in merged_fg_diff:
        if len(diff_tuple) >= 3:
            fg_atoms = diff_tuple[2]  # [[atom1, atom2], [atom3, atom4], ...] or [[[atom1]]]
            if isinstance(fg_atoms, list):
                for atom_list in fg_atoms:
                    if isinstance(atom_list, (list, tuple)):
                        # Check if it's nested (e.g., [[11]])
                        if len(atom_list) > 0 and isinstance(atom_list[0], (list, tuple)):
                            # Double nested: [[[11]]] -> [[11]] -> [11]
                            for inner_list in atom_list:
                                if isinstance(inner_list, (list, tuple)):
                                    atoms_to_remove.update(inner_list)
                                elif isinstance(inner_list, int):
                                    atoms_to_remove.add(inner_list)
                        else:
                            # Single level: [[11], [13]] -> [11] or [13]
                            atoms_to_remove.update(atom_list)
                    elif isinstance(atom_list, int):
                        atoms_to_remove.add(atom_list)
    
    if not atoms_to_remove:
        return smiles
    
    # Remove atoms
    remain_mol = remove_atoms_from_mol(mol, atoms_to_remove)
    if not remain_mol:
        return None
    
    try:
        return Chem.MolToSmiles(remain_mol, isomericSmiles=False)
    except:
        return None


def visualize_pair(
    row: pd.Series,
    output_path: Path,
    pair_type: str,
    pair_idx: int
) -> None:
    """Visualize a single pair with 4 molecules.
    
    Args:
        row: DataFrame row with pair information
        output_path: Directory to save visualization
        pair_type: 'valid' or 'invalid'
        pair_idx: Index of the pair
    """
    # Parse data
    toxic_smiles = row['toxic_smiles']
    nontoxic_smiles = row['nontoxic_smiles']
    toxic_canonical = row.get('toxic_canonical_smiles', toxic_smiles)
    nontoxic_canonical = row.get('nontoxic_canonical_smiles', nontoxic_smiles)
    
    # Parse FG full
    toxic_fg_full = parse_fg_full(row.get('toxic_fg_full', '{}'))
    nontoxic_fg_full = parse_fg_full(row.get('nontoxic_fg_full', '{}'))
    
    # Parse unique_fg
    unique_fg_str = row.get('unique_fg', '[]')
    try:
        if isinstance(unique_fg_str, str):
            unique_fg_list = ast.literal_eval(unique_fg_str)
        else:
            unique_fg_list = unique_fg_str
    except:
        unique_fg_list = []
    
    # Get diff tuples
    toxic_diff = get_unique_fg_diff_tuples(
        unique_fg_list, toxic_fg_full, nontoxic_fg_full, 'name_only_in_toxic'
    )
    nontoxic_diff = get_unique_fg_diff_tuples(
        unique_fg_list, toxic_fg_full, nontoxic_fg_full, 'name_only_in_nontoxic'
    )
    
    # Handle atom_index_diff
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
        
        toxic_atom_indices = fg_info.get('toxic_atom_indices', [])
        for atom_item in toxic_atom_indices:
            if isinstance(atom_item, (list, tuple)):
                fg_atoms = list(atom_item) if isinstance(atom_item, tuple) else atom_item
                # fg_atoms is already a list, wrap it in another list for AccFG format
                atom_diff_toxic.append((fg_name, fg_name, [fg_atoms]))
            elif isinstance(atom_item, int):
                # Single atom, wrap in list then in another list
                atom_diff_toxic.append((fg_name, fg_name, [[atom_item]]))
        
        nontoxic_atom_indices = fg_info.get('nontoxic_atom_indices', [])
        for atom_item in nontoxic_atom_indices:
            if isinstance(atom_item, (list, tuple)):
                fg_atoms = list(atom_item) if isinstance(atom_item, tuple) else atom_item
                atom_diff_nontoxic.append((fg_name, fg_name, [fg_atoms]))
            elif isinstance(atom_item, int):
                atom_diff_nontoxic.append((fg_name, fg_name, [[atom_item]]))
    
    target_diff = merge_diff_tuple(toxic_diff + atom_diff_toxic)
    ref_diff = merge_diff_tuple(nontoxic_diff + atom_diff_nontoxic)
    
    # Get removed SMILES
    toxic_removed = get_removed_smiles(toxic_canonical, target_diff, toxic_fg_full)
    nontoxic_removed = get_removed_smiles(nontoxic_canonical, ref_diff, nontoxic_fg_full)
    
    # Extract atom indices to highlight (unique FGs that will be removed)
    # These indices are from unique_fg and correspond to atomNote indices
    def extract_atom_indices_from_diff(diff_list):
        """Extract flat list of atom indices from diff tuples."""
        atoms = set()
        for diff_tuple in diff_list:
            if len(diff_tuple) >= 3:
                fg_atoms = diff_tuple[2]  # [[atom1, atom2], [atom3, atom4], ...] or [[[atom1]]]
                if isinstance(fg_atoms, list):
                    for atom_list in fg_atoms:
                        if isinstance(atom_list, (list, tuple)):
                            # Check if it's nested (e.g., [[11]])
                            if len(atom_list) > 0 and isinstance(atom_list[0], (list, tuple)):
                                # Double nested: [[[11]]] -> [[11]] -> [11]
                                for inner_list in atom_list:
                                    if isinstance(inner_list, (list, tuple)):
                                        atoms.update(inner_list)
                                    elif isinstance(inner_list, int):
                                        atoms.add(inner_list)
                            else:
                                # Single level: [[11], [13]] -> [11] or [13]
                                atoms.update(atom_list)
                        elif isinstance(atom_list, int):
                            atoms.add(atom_list)
        return atoms
    
    toxic_atom_indices = extract_atom_indices_from_diff(target_diff)
    nontoxic_atom_indices = extract_atom_indices_from_diff(ref_diff)
    
    # Create molecules with highlight information
    mols = []
    labels = []
    smiles_list = []
    highlight_atoms_list = []
    
    # 1. Toxic original (with highlight)
    mol_toxic = Chem.MolFromSmiles(toxic_smiles)
    if mol_toxic:
        # Set atom indices using atomNote (same as in exam_comparison)
        if ACCFG_AVAILABLE and set_atom_idx:
            try:
                mol_toxic = set_atom_idx(mol_toxic, 'atomNote')
            except:
                pass
        
        # Map atomNote indices to RDKit atom indices
        # After set_atom_idx, atomNote should match GetIdx()
        valid_toxic_atoms = []
        for atom in mol_toxic.GetAtoms():
            atom_idx = atom.GetIdx()
            # Check if this atom has atomNote matching our target indices
            if atom.HasProp('atomNote'):
                atom_note_val = int(atom.GetProp('atomNote'))
                if atom_note_val in toxic_atom_indices:
                    valid_toxic_atoms.append(atom_idx)
            elif atom_idx in toxic_atom_indices:
                # Fallback: use GetIdx() directly if atomNote not set
                valid_toxic_atoms.append(atom_idx)
        
        mols.append(mol_toxic)
        labels.append("Toxic (Original)")
        smiles_list.append(toxic_smiles)
        highlight_atoms_list.append(valid_toxic_atoms)
    
    # 2. Nontoxic original (with highlight)
    mol_nontoxic = Chem.MolFromSmiles(nontoxic_smiles)
    if mol_nontoxic:
        # Set atom indices using atomNote (same as in exam_comparison)
        if ACCFG_AVAILABLE and set_atom_idx:
            try:
                mol_nontoxic = set_atom_idx(mol_nontoxic, 'atomNote')
            except:
                pass
        
        # Map atomNote indices to RDKit atom indices
        valid_nontoxic_atoms = []
        for atom in mol_nontoxic.GetAtoms():
            atom_idx = atom.GetIdx()
            # Check if this atom has atomNote matching our target indices
            if atom.HasProp('atomNote'):
                atom_note_val = int(atom.GetProp('atomNote'))
                if atom_note_val in nontoxic_atom_indices:
                    valid_nontoxic_atoms.append(atom_idx)
            elif atom_idx in nontoxic_atom_indices:
                # Fallback: use GetIdx() directly if atomNote not set
                valid_nontoxic_atoms.append(atom_idx)
        
        mols.append(mol_nontoxic)
        labels.append("Nontoxic (Original)")
        smiles_list.append(nontoxic_smiles)
        highlight_atoms_list.append(valid_nontoxic_atoms)
    
    # 3. Toxic after removal (no highlight)
    if toxic_removed:
        mol_toxic_removed = Chem.MolFromSmiles(toxic_removed)
        if mol_toxic_removed:
            mols.append(mol_toxic_removed)
            labels.append("Toxic (FG Removed)")
            smiles_list.append(toxic_removed)
            highlight_atoms_list.append([])
    
    # 4. Nontoxic after removal (no highlight)
    if nontoxic_removed:
        mol_nontoxic_removed = Chem.MolFromSmiles(nontoxic_removed)
        if mol_nontoxic_removed:
            mols.append(mol_nontoxic_removed)
            labels.append("Nontoxic (FG Removed)")
            smiles_list.append(nontoxic_removed)
            highlight_atoms_list.append([])
    
    if len(mols) < 2:
        print(f"⚠️  Skipping pair {pair_idx}: Not enough valid molecules")
        return
    
    # Create visualization using PIL Images directly
    images = []
    for i, (mol, label, smiles) in enumerate(zip(mols, labels, smiles_list)):
        highlight_atoms = highlight_atoms_list[i] if i < len(highlight_atoms_list) else []
        
        # Draw molecule with highlights for original molecules
        if highlight_atoms:
            # Use different colors for toxic (red) and nontoxic (blue)
            if "Toxic" in label:
                highlight_color = (1.0, 0.7, 0.7)  # Light red
            else:
                highlight_color = (0.7, 0.7, 1.0)  # Light blue
            
            img = Draw.MolToImage(
                mol, 
                size=(500, 500),
                highlightAtoms=highlight_atoms,
                highlightColor=highlight_color
            )
        else:
            img = Draw.MolToImage(mol, size=(500, 500))
        images.append((img, label, smiles))
    
    # Calculate dimensions
    mol_width = 500
    mol_height = 500
    text_height = 150  # Space for label and SMILES (increased)
    padding = 30  # Padding between images (increased)
    top_margin = 20  # Top margin
    
    total_width = len(images) * (mol_width + padding) - padding
    total_height = mol_height + text_height + top_margin + 30
    
    # Create combined image with text
    combined = Image.new('RGB', (total_width, total_height), 'white')
    
    # Add text (using PIL's ImageDraw)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    
    # Try to load a nice font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        smiles_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 22)
            smiles_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            smiles_font = ImageFont.load_default()
    
    x_offset = 0
    for img, label, smiles in images:
        # Paste molecule image (below text area)
        mol_y = top_margin + text_height
        combined.paste(img, (x_offset, mol_y))
        
        # Add label (title) - bold and clear
        draw.text((x_offset + 15, top_margin + 5), label, fill='black', font=title_font)
        
        # Add SMILES string (split into multiple lines if too long)
        max_chars_per_line = 65
        if len(smiles) > max_chars_per_line:
            # Split SMILES into multiple lines
            lines = []
            for i in range(0, len(smiles), max_chars_per_line):
                lines.append(smiles[i:i+max_chars_per_line])
            smiles_lines = lines
        else:
            smiles_lines = [smiles]
        
        # Draw each line of SMILES
        y_text = top_margin + 35
        for line in smiles_lines:
            draw.text((x_offset + 15, y_text), line, fill='darkblue', font=smiles_font)
            y_text += 18  # Line spacing
        
        # Add vertical separator line
        if x_offset > 0:
            draw.line([(x_offset - padding//2, top_margin), 
                      (x_offset - padding//2, total_height - 10)], 
                     fill='lightgray', width=2)
        
        x_offset += mol_width + padding
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{pair_type}_pair_{pair_idx}.png"
    combined.save(output_file)
    
    print(f"✅ Saved: {output_file}")
    print(f"   Toxic original: {toxic_smiles}")
    print(f"   Nontoxic original: {nontoxic_smiles}")
    if toxic_removed:
        print(f"   Toxic removed: {toxic_removed}")
    if nontoxic_removed:
        print(f"   Nontoxic removed: {nontoxic_removed}")
    print()


def main():
    """Visualize sample pairs from valid and invalid CSVs."""
    # Paths
    valid_csv = ROOT / "molecular_feature" / "pairs_fg_stereo_merged_nodot_validated.csv"
    invalid_csv = ROOT / "molecular_feature" / "pairs_fg_stereo_merged_nodot_validated_invalid.csv"
    output_dir = ROOT / "molecular_feature" / "visualizations"
    
    # Load data
    print("Loading valid pairs...")
    df_valid = pd.read_csv(valid_csv, low_memory=False)
    print(f"  Total valid pairs: {len(df_valid)}")
    
    print("Loading invalid pairs...")
    df_invalid_raw = pd.read_csv(invalid_csv, low_memory=False)
    print(f"  Total invalid pairs: {len(df_invalid_raw)}")
    
    # Parse invalid pairs - they have 'original_row' as a string dict
    if 'original_row' in df_invalid_raw.columns:
        import ast
        import numpy as np
        invalid_rows = []
        for idx, row in df_invalid_raw.iterrows():
            try:
                original_val = row.get('original_row')
                if pd.isna(original_val) or str(original_val).strip() == '' or str(original_val).lower() == 'nan':
                    continue
                
                original_str = str(original_val)
                # Replace 'nan' with None for proper parsing
                original_str = original_str.replace('nan', 'None')
                # Use ast.literal_eval (safer than eval)
                original_dict = ast.literal_eval(original_str)
                # Convert None back to np.nan for pandas compatibility
                for key, val in original_dict.items():
                    if val is None:
                        original_dict[key] = np.nan
                invalid_rows.append(original_dict)
            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"  ⚠️  Error parsing invalid row {idx}: {e}")
                continue
        if invalid_rows:
            df_invalid = pd.DataFrame(invalid_rows)
            print(f"  Parsed {len(df_invalid)} invalid pairs")
        else:
            print("  ⚠️  No invalid pairs could be parsed")
            df_invalid = pd.DataFrame()
    else:
        df_invalid = df_invalid_raw
    
    # Sample 3 from each
    n_samples = 3
    valid_samples = df_valid.head(n_samples)
    invalid_samples = df_invalid.head(n_samples)
    
    print(f"\n📊 Visualizing {n_samples} valid pairs...")
    for idx, (_, row) in enumerate(valid_samples.iterrows(), 1):
        visualize_pair(row, output_dir, 'valid', idx)
    
    print(f"\n📊 Visualizing {n_samples} invalid pairs...")
    for idx, (_, row) in enumerate(invalid_samples.iterrows(), 1):
        visualize_pair(row, output_dir, 'invalid', idx)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
