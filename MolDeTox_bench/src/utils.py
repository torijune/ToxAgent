"""Common utility functions for QA generation.

All QA generation is self-contained within MolDeTox_bench/src/.
No dependency on Mol_FG or Mol_stereo is required.
"""
import ast
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# RDKit availability check
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ===========================================================================
# MolecularConverter  (previously in Mol_FG/qa_builder.py and
#                       Mol_stereo/stereo_qa_builder.py)
# ===========================================================================

class MolecularConverter:
    """Convert SMILES to the requested molecular format (SMILES or SELFIES)."""

    def __init__(self, format_type: str = "smiles"):
        self.format_type = format_type.lower()
        if self.format_type == "selfies":
            try:
                import selfies as _sf  # noqa: F401
            except ImportError:
                raise ImportError(
                    "SELFIES library not available. Install with: pip install selfies"
                )

    def convert(self, smiles: str) -> str:
        """Convert SMILES to the configured format."""
        if not smiles or smiles in ("NA", "nan"):
            return smiles
        if self.format_type == "selfies":
            try:
                import selfies as sf
                return sf.encoder(smiles)
            except Exception as e:
                print(f"Warning: Failed to convert '{smiles}' to SELFIES: {e}")
                return smiles
        return smiles  # SMILES → SMILES is identity


# ===========================================================================
# add_atom_numbers_to_smiles  (previously in Mol_FG/utils.py and
#                               Mol_stereo/stereo_utils.py)
# ===========================================================================

def add_atom_numbers_to_smiles(smiles: str) -> Optional[str]:
    """Add atom map numbers to every atom in a SMILES string.

    Atom indices are 0-based (atom 0, 1, 2, …).
    Example: "CN"  →  "[C:0][N:1]"

    Returns:
        SMILES with atom map numbers, or the original SMILES if RDKit is
        unavailable or parsing fails.
    """
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(idx)
        return Chem.MolToSmiles(mol, canonical=False)
    except Exception:
        return smiles


# ===========================================================================
# explain_stereo_for_repair  (previously in Mol_stereo/stereo_extractor.py
#                              as explain_stereochemistry_for_repair)
# ===========================================================================

def explain_stereo_for_repair(
    comparison: Dict[str, Any],
    add_atom_indices: bool = True,
) -> str:
    """Generate a Remove/Add-style description of stereochemistry differences.

    Args:
        comparison: dict with keys 'toxic_stereochemistry',
                    'nontoxic_stereochemistry', 'stereochemistry_difference'.
        add_atom_indices: Whether to include atom indices in the output.

    Returns:
        Multi-line natural-language string describing the stereo changes.
    """
    toxic_stereo = comparison.get("toxic_stereochemistry") or {}
    nontoxic_stereo = comparison.get("nontoxic_stereochemistry") or {}

    has_tx = toxic_stereo.get("has_stereochemistry", False)
    has_nt = nontoxic_stereo.get("has_stereochemistry", False)
    if not has_tx and not has_nt:
        return "Neither molecule contains stereochemical information."

    # ---- helpers ----
    def _chiral_map(stereo: Dict) -> Dict[Any, Dict]:
        centers = stereo.get("chiral_centers", {}).get("chiral_centers", []) or []
        return {c["atom_idx"]: c for c in centers if isinstance(c, dict) and "atom_idx" in c}

    def _ez_map(stereo: Dict) -> Dict[Any, Dict]:
        bonds = stereo.get("ez_bonds", {}).get("ez_bonds", []) or []
        result = {}
        for b in bonds:
            if not isinstance(b, dict):
                continue
            key = tuple(b["atoms"]) if isinstance(b.get("atoms"), list) else b.get("atoms")
            result[key] = b
        return result

    def _chirality_label(detail: Dict) -> str:
        raw = detail.get("chirality", detail.get("config", "?"))
        return str(raw).split("/")[0] if raw else "?"

    def _atom_sym(detail: Dict) -> str:
        return detail.get("atom_symbol", "?")

    tx_chiral = _chiral_map(toxic_stereo)
    nt_chiral = _chiral_map(nontoxic_stereo)
    tx_ez = _ez_map(toxic_stereo)
    nt_ez = _ez_map(nontoxic_stereo)

    removing, adding = [], []

    # Chiral: in toxic but different / missing in nontoxic → remove
    for atom_idx, td in tx_chiral.items():
        nd = nt_chiral.get(atom_idx)
        if nd is None or _chirality_label(td) != _chirality_label(nd):
            cfg = _chirality_label(td)
            sym = _atom_sym(td)
            if add_atom_indices:
                removing.append(f"  - Chiral center at Atom {atom_idx} ({sym}): {cfg}-configuration")
            else:
                removing.append(f"  - Chiral center ({sym}): {cfg}-configuration")

    # Chiral: in nontoxic but different / missing in toxic → add
    for atom_idx, nd in nt_chiral.items():
        td = tx_chiral.get(atom_idx)
        if td is None or _chirality_label(nd) != _chirality_label(td):
            cfg = _chirality_label(nd)
            sym = _atom_sym(nd)
            if add_atom_indices:
                adding.append(f"  - Chiral center at Atom {atom_idx} ({sym}): {cfg}-configuration")
            else:
                adding.append(f"  - Chiral center ({sym}): {cfg}-configuration")

    # E/Z: in toxic but different / missing in nontoxic → remove
    for key, td in tx_ez.items():
        nd = nt_ez.get(key)
        if nd is None or td.get("geometry") != nd.get("geometry"):
            atoms = list(key) if isinstance(key, tuple) else key
            syms = td.get("atom_symbols", ["?", "?"])
            geom = td.get("geometry", "?")
            if add_atom_indices and isinstance(atoms, list) and len(atoms) >= 2:
                removing.append(
                    f"  - E/Z bond at Atoms {atoms[0]}-{atoms[1]} "
                    f"({syms[0]}={syms[1]}): {geom}-geometry"
                )
            else:
                removing.append(f"  - E/Z bond ({syms[0]}={syms[1]}): {geom}-geometry")

    # E/Z: in nontoxic but different / missing in toxic → add
    for key, nd in nt_ez.items():
        td = tx_ez.get(key)
        if td is None or nd.get("geometry") != td.get("geometry"):
            atoms = list(key) if isinstance(key, tuple) else key
            syms = nd.get("atom_symbols", ["?", "?"])
            geom = nd.get("geometry", "?")
            if add_atom_indices and isinstance(atoms, list) and len(atoms) >= 2:
                adding.append(
                    f"  - E/Z bond at Atoms {atoms[0]}-{atoms[1]} "
                    f"({syms[0]}={syms[1]}): {geom}-geometry"
                )
            else:
                adding.append(f"  - E/Z bond ({syms[0]}={syms[1]}): {geom}-geometry")

    if not removing and not adding:
        return "No significant stereochemical changes required."

    lines = []
    if removing:
        lines.append("Removing Toxic Stereochemical Features:\n")
        lines.extend(removing)
        lines.append("")
    if adding:
        lines.append("Adding Non-Toxic Stereochemical Features:\n")
        lines.extend(adding)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward-compat alias (old code used this name)
# ---------------------------------------------------------------------------
explain_stereochemistry_for_repair = explain_stereo_for_repair


def parse_fg_full(fg_full_str: str) -> Dict[str, List[tuple]]:
    """Parse toxic_fg_full or nontoxic_fg_full string to dictionary.
    
    Args:
        fg_full_str: String representation of FG full dictionary
        
    Returns:
        Dictionary mapping FG names to lists of atom index tuples
    """
    if pd.isna(fg_full_str) or not fg_full_str or fg_full_str == "NA":
        return {}
    try:
        # Try to parse as Python dict literal
        fg_dict = ast.literal_eval(fg_full_str)
        if isinstance(fg_dict, dict):
            # Convert list of lists to list of tuples
            result = {}
            for fg_name, atom_indices_list in fg_dict.items():
                if isinstance(atom_indices_list, list):
                    result[fg_name] = [tuple(atom_indices) if isinstance(atom_indices, list) else atom_indices 
                                     for atom_indices in atom_indices_list]
                else:
                    result[fg_name] = [atom_indices_list]
            return result
    except Exception as e:
        print(f"Warning: Failed to parse fg_full: {e}")
    return {}


def parse_unique_fg(unique_fg_str: str) -> List[Dict[str, Any]]:
    """Parse unique_fg JSON string to list of dictionaries.
    
    Args:
        unique_fg_str: String representation of unique FG list
        
    Returns:
        List of dictionaries containing FG information
    """
    if pd.isna(unique_fg_str) or not unique_fg_str or unique_fg_str == "NA":
        return []
    try:
        unique_fg_list = ast.literal_eval(unique_fg_str)
        if isinstance(unique_fg_list, list):
            return unique_fg_list
    except Exception as e:
        print(f"Warning: Failed to parse unique_fg: {e}")
    return []


def safe_get_value(row: pd.Series, keys: List[str], default: Any = None) -> Any:
    """Safely get value from pandas Series with fallback keys.
    
    Args:
        row: Pandas Series to extract value from
        keys: List of keys to try in order
        default: Default value if none of the keys exist
        
    Returns:
        Value from the first matching key, or default
    """
    for key in keys:
        if key in row:
            return row[key]
    return default


def filter_csv_by_type(
    df: pd.DataFrame,
    diff_type: Optional[str] = None,
    step_num: Optional[str] = None
) -> pd.DataFrame:
    """Filter DataFrame by diff_type and/or step_num columns.
    
    This function supports both the old filtering method (n_fg_diff, n_stereo_diff, n_diff_features)
    and the new method (diff_type, step_num columns).
    
    Args:
        df: Input DataFrame
        diff_type: Filter by diff_type ('only_fg', 'only_stereo', 'both', or None for all)
        step_num: Filter by step_num ('one_step', 'multi_step', or None for all)
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Use new columns if available, otherwise fall back to old method
    if 'diff_type' in df.columns:
        if diff_type:
            filtered_df = filtered_df[filtered_df['diff_type'] == diff_type]
    else:
        # Fall back to old filtering method (only if columns exist)
        if diff_type == 'only_fg':
            if 'n_fg_diff' in df.columns and 'n_stereo_diff' in df.columns:
                filtered_df = filtered_df[(filtered_df['n_fg_diff'] > 0) & (filtered_df['n_stereo_diff'] == 0)]
            else:
                raise ValueError("Cannot filter by diff_type='only_fg': required columns 'diff_type' or 'n_fg_diff'/'n_stereo_diff' not found")
        elif diff_type == 'only_stereo':
            if 'n_fg_diff' in df.columns and 'n_stereo_diff' in df.columns:
                filtered_df = filtered_df[(filtered_df['n_fg_diff'] == 0) & (filtered_df['n_stereo_diff'] > 0)]
            else:
                raise ValueError("Cannot filter by diff_type='only_stereo': required columns 'diff_type' or 'n_fg_diff'/'n_stereo_diff' not found")
        elif diff_type == 'both':
            if 'n_fg_diff' in df.columns and 'n_stereo_diff' in df.columns:
                filtered_df = filtered_df[(filtered_df['n_fg_diff'] > 0) & (filtered_df['n_stereo_diff'] > 0)]
            else:
                raise ValueError("Cannot filter by diff_type='both': required columns 'diff_type' or 'n_fg_diff'/'n_stereo_diff' not found")
    
    if 'step_num' in df.columns:
        if step_num:
            filtered_df = filtered_df[filtered_df['step_num'] == step_num]
    else:
        # Fall back to old filtering method (only if column exists)
        if step_num == 'one_step':
            if 'n_diff_features' in df.columns:
                filtered_df = filtered_df[filtered_df['n_diff_features'] == 1]
            else:
                raise ValueError("Cannot filter by step_num='one_step': required columns 'step_num' or 'n_diff_features' not found")
        elif step_num == 'multi_step':
            if 'n_diff_features' in df.columns:
                filtered_df = filtered_df[filtered_df['n_diff_features'] > 1]
            else:
                raise ValueError("Cannot filter by step_num='multi_step': required columns 'step_num' or 'n_diff_features' not found")
    
    return filtered_df


def parse_stereo_dict(stereo_str: Any) -> Dict[str, Any]:
    """Parse stereochemistry dictionary from string or dict.
    
    Args:
        stereo_str: String representation of stereochemistry dict, or dict itself
        
    Returns:
        Dictionary with stereochemistry information, or empty dict if parsing fails
    """
    if pd.isna(stereo_str) or not stereo_str or stereo_str == "NA":
        return {}
    try:
        if isinstance(stereo_str, dict):
            return stereo_str
        if isinstance(stereo_str, str):
            # Try JSON first
            try:
                return json.loads(stereo_str)
            except:
                # Try ast.literal_eval
                return ast.literal_eval(stereo_str)
    except Exception as e:
        print(f"Warning: Failed to parse stereochemistry dict: {e}")
    return {}


def extract_smiles(row: pd.Series, toxic_key: str = 'toxic_canonical_smiles', 
                   nontoxic_key: str = 'nontoxic_canonical_smiles') -> Tuple[Optional[str], Optional[str]]:
    """Extract and validate SMILES strings from DataFrame row.
    
    Args:
        row: Pandas Series row
        toxic_key: Column name for toxic SMILES
        nontoxic_key: Column name for non-toxic SMILES
        
    Returns:
        Tuple of (toxic_smiles, nontoxic_smiles), or (None, None) if invalid
    """
    tx_smiles = str(row[toxic_key]).strip() if toxic_key in row else None
    nt_smiles = str(row[nontoxic_key]).strip() if nontoxic_key in row else None
    
    if (not tx_smiles or not nt_smiles or 
        pd.isna(tx_smiles) or pd.isna(nt_smiles) or 
        tx_smiles == "NA" or nt_smiles == "NA"):
        return None, None
    
    return tx_smiles, nt_smiles


def parse_fg_list(fg_str: Any) -> List[str]:
    """Parse functional group list from string or list.
    
    Args:
        fg_str: String representation of FG list, or list itself
        
    Returns:
        List of functional group names
    """
    if pd.isna(fg_str) or not fg_str or fg_str == "NA":
        return []
    try:
        if isinstance(fg_str, list):
            return fg_str
        if isinstance(fg_str, str):
            fg_list = ast.literal_eval(fg_str)
            if isinstance(fg_list, list):
                return fg_list
            elif fg_list:
                return [fg_list]
    except Exception as e:
        print(f"Warning: Failed to parse FG list: {e}")
    return []


def extract_dataset_info(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """Extract dataset name and endpoint from DataFrame row.
    
    Args:
        row: Pandas Series row
        
    Returns:
        Tuple of (dataset_name, endpoint)
    """
    dataset_name = None
    endpoint = None
    
    if 'dataset_name' in row and not pd.isna(row['dataset_name']):
        dataset_name = str(row['dataset_name']).strip()
    
    if 'endpoint' in row and not pd.isna(row['endpoint']):
        endpoint = str(row['endpoint']).strip()
    
    return dataset_name, endpoint


def calculate_unique_fgs(tx_fgs: List[str], nt_fgs: List[str]) -> Tuple[List[str], List[str]]:
    """Calculate unique functional groups between toxic and non-toxic molecules.
    
    Args:
        tx_fgs: List of toxic functional groups
        nt_fgs: List of non-toxic functional groups
        
    Returns:
        Tuple of (toxic_unique_fgs, nontoxic_unique_fgs)
    """
    tx_unique = sorted(set(tx_fgs) - set(nt_fgs))
    nt_unique = sorted(set(nt_fgs) - set(tx_fgs))
    return tx_unique, nt_unique


def save_qa_items_to_jsonl(qa_items: List[Dict[str, Any]], output_path: str) -> None:
    """Save QA items to JSONL file.
    
    Args:
        qa_items: List of QA item dictionaries
        output_path: Path to save JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in qa_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(qa_items)} QA items to {output_path}")


def create_base_context(row: pd.Series, task_type: str, 
                       tx_smiles: Optional[str] = None,
                       nt_smiles: Optional[str] = None) -> Dict[str, Any]:
    """Create base context dictionary for QA items.
    
    Args:
        row: Pandas Series row
        task_type: Type of task (e.g., 'fg_identification')
        tx_smiles: Optional toxic SMILES to include
        nt_smiles: Optional non-toxic SMILES to include
        
    Returns:
        Base context dictionary
    """
    dataset_name, endpoint = extract_dataset_info(row)
    
    context = {
        "task_type": task_type,
        "dataset": dataset_name,
        "endpoint": endpoint,
    }
    
    if tx_smiles:
        context["toxic_smiles"] = tx_smiles
    if nt_smiles:
        context["nontoxic_smiles"] = nt_smiles
    
    # Add similarity if available
    if 'tanimoto_sim' in row and not pd.isna(row['tanimoto_sim']):
        try:
            context["similarity"] = float(row['tanimoto_sim'])
        except:
            pass
    
    return context


def setup_project_imports(file_path: str) -> Tuple[Path, Path]:
    """(Deprecated) Setup import paths — no longer needed.

    QA generation is now fully self-contained within MolDeTox_bench/src/.
    Kept for backward compatibility only.
    """
    current_file = Path(file_path)
    project_root = current_file.parent.parent.parent.parent.parent
    src_dir = current_file.parent.parent.parent.parent
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return project_root, src_dir


def import_fg_components(project_root: Path):
    """(Deprecated) Returns OnlyFGQATemplate and MolecularConverter from QA_templates."""
    from QA_templates.only_fg import OnlyFGQATemplate
    return OnlyFGQATemplate, MolecularConverter


def import_stereo_components(project_root: Path):
    """(Deprecated) Returns OnlyStereoQATemplate and MolecularConverter from QA_templates."""
    from QA_templates.only_stereo import OnlyStereoQATemplate
    return OnlyStereoQATemplate, MolecularConverter


def load_and_filter_csv(
    csv_path: str,
    diff_type: Optional[str] = None,
    step_num: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Load CSV file and filter by diff_type and step_num.
    
    Args:
        csv_path: Path to CSV file
        diff_type: Filter by diff_type ('only_fg', 'only_stereo', 'both', or None)
        step_num: Filter by step_num ('one_step', 'multi_step', or None)
        verbose: Whether to print loading messages
        
    Returns:
        Filtered DataFrame
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If DataFrame is empty after filtering
    """
    if verbose:
        print(f"Loading pairs from {csv_path}...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    if verbose:
        print(f"Total pairs: {len(df)}")
    
    # Filter if needed
    if diff_type or step_num:
        df = filter_csv_by_type(df, diff_type=diff_type, step_num=step_num)
        if verbose:
            filter_desc = []
            if diff_type:
                filter_desc.append(diff_type)
            if step_num:
                filter_desc.append(step_num)
            print(f"Filtered to {' + '.join(filter_desc)}: {len(df)}")
    
    if len(df) == 0:
        filter_desc = []
        if diff_type:
            filter_desc.append(diff_type)
        if step_num:
            filter_desc.append(step_num)
        desc = ' + '.join(filter_desc) if filter_desc else 'specified criteria'
        if verbose:
            print(f"No pairs found with {desc}.")
        raise ValueError(f"No pairs found with {desc}")
    
    return df


def initialize_qa_components(smiles_format: str, template_class, converter_class):
    """Initialize QA generation components.
    
    Args:
        smiles_format: Molecular format ('smiles' or 'selfies')
        template_class: Template class (QATemplate or StereoQATemplate)
        converter_class: MolecularConverter class
        
    Returns:
        Tuple of (converter, template, mol_format)
    """
    converter = converter_class(smiles_format)
    template = template_class()
    mol_format = "SELFIES" if smiles_format == "selfies" else "SMILES"
    return converter, template, mol_format


def load_check_diff_csv(
    csv_path: str,
    diff_type: str,
    step_num: str
) -> pd.DataFrame:
    """Load pre-filtered check_diff CSV. No further filtering is needed.
    
    If csv_path is a directory, automatically constructs the filename as
    {diff_type}_{step_num}_diff.csv. If csv_path is a file, loads it directly.
    
    Args:
        csv_path: Path to check_diff directory OR specific CSV file
        diff_type: 'only_fg', 'only_stereo', or 'both'
        step_num: 'one_step' or 'multi_step'
        
    Returns:
        Loaded DataFrame (already filtered, no further processing needed)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    path = Path(csv_path)
    
    if path.is_dir():
        csv_file = path / f"{diff_type}_{step_num}_diff.csv"
    else:
        csv_file = path
    
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Pre-filtered CSV not found: {csv_file}\n"
            f"Generate it first by running: molecular_feature/check_diff/{diff_type}_diff.py"
        )
    
    print(f"Loading pre-filtered pairs from: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"Total pairs: {len(df)}")
    return df


def extract_fg_uniques(row: pd.Series) -> Tuple[List[str], List[str], List[str]]:
    """Read pre-extracted FG unique columns from check_diff CSV row.
    
    These columns are pre-computed by only_fg_diff.py and stored in:
    - toxic_unique_fg_names: FGs only in toxic molecule (name_only_in_toxic)
    - nontoxic_unique_fg_names: FGs only in nontoxic molecule (name_only_in_nontoxic)
    - atom_diff_fg_names: FGs present in both but at different atom positions (atom_index_diff)
    
    Args:
        row: DataFrame row from check_diff CSV
        
    Returns:
        Tuple of (tx_unique, nt_unique, atom_diff_fgs)
    """
    tx_unique = parse_fg_list(row.get('toxic_unique_fg_names', '[]'))
    nt_unique = parse_fg_list(row.get('nontoxic_unique_fg_names', '[]'))
    atom_diff_fgs = parse_fg_list(row.get('atom_diff_fg_names', '[]'))
    return tx_unique, nt_unique, atom_diff_fgs


def extract_stereo_uniques(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Read pre-extracted stereo unique columns from check_diff CSV row.
    
    These columns are pre-computed by only_stereo_diff.py and stored in:
    - toxic_unique_stereo: stereochemistry features unique to the toxic molecule
    - nontoxic_unique_stereo: stereochemistry features unique to the nontoxic molecule
    
    Args:
        row: DataFrame row from check_diff CSV
        
    Returns:
        Tuple of (toxic_unique_stereo, nontoxic_unique_stereo) as lists of strings
    """
    tx_unique = parse_fg_list(row.get('toxic_unique_stereo', '[]'))
    nt_unique = parse_fg_list(row.get('nontoxic_unique_stereo', '[]'))
    return tx_unique, nt_unique


def build_stereo_uniques_from_columns(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Build toxic/nontoxic unique stereo by comparing individual chiral/ez columns.
    
    Used as fallback when pre-extracted columns (toxic_unique_stereo) are empty.
    Covers cases like: nontoxic has chiral centers that toxic doesn't, etc.
    
    Args:
        row: DataFrame row from check_diff CSV
        
    Returns:
        Tuple of (toxic_unique_stereo, nontoxic_unique_stereo) as lists of strings
    """
    toxic_unique: List[str] = []
    nontoxic_unique: List[str] = []

    # Parse chiral centers
    tx_chiral_raw = parse_fg_list(row.get('toxic_chiral_centers', '[]'))
    nt_chiral_raw = parse_fg_list(row.get('nontoxic_chiral_centers', '[]'))

    tx_map = {c['atom_idx']: c.get('config', '') for c in tx_chiral_raw
              if isinstance(c, dict) and 'atom_idx' in c}
    nt_map = {c['atom_idx']: c.get('config', '') for c in nt_chiral_raw
              if isinstance(c, dict) and 'atom_idx' in c}

    # Toxic-unique chiral centers
    for atom_idx, config in tx_map.items():
        if atom_idx not in nt_map or nt_map[atom_idx] != config:
            toxic_unique.append(f"Chiral center at atom {atom_idx} ({config} configuration)")

    # Nontoxic-unique chiral centers (toxic lacks these)
    for atom_idx, config in nt_map.items():
        if atom_idx not in tx_map or tx_map[atom_idx] != config:
            nontoxic_unique.append(f"Chiral center at atom {atom_idx} ({config} configuration)")

    # Parse E/Z bonds
    tx_ez_raw = parse_fg_list(row.get('toxic_ez_bonds', '[]'))
    nt_ez_raw = parse_fg_list(row.get('nontoxic_ez_bonds', '[]'))

    def _ez_key(b: Any) -> Any:
        bond = b.get('bond', '')
        return tuple(bond) if isinstance(bond, list) else bond

    tx_ez_map = {_ez_key(b): b.get('geometry', '') for b in tx_ez_raw if isinstance(b, dict)}
    nt_ez_map = {_ez_key(b): b.get('geometry', '') for b in nt_ez_raw if isinstance(b, dict)}

    for bond, geometry in tx_ez_map.items():
        if bond not in nt_ez_map or nt_ez_map[bond] != geometry:
            toxic_unique.append(f"E/Z bond {list(bond) if isinstance(bond, tuple) else bond}: {geometry} geometry")

    for bond, geometry in nt_ez_map.items():
        if bond not in tx_ez_map or tx_ez_map[bond] != geometry:
            nontoxic_unique.append(f"E/Z bond {list(bond) if isinstance(bond, tuple) else bond}: {geometry} geometry")

    # Fallback: use diff type columns if still nothing found
    if not toxic_unique and not nontoxic_unique:
        chiral_diff = row.get('chiral_diff_loose', False)
        ez_diff = row.get('ez_diff_loose', False)
        diff_type = str(row.get('stereo_diff_type_loose', '')).strip()
        if chiral_diff:
            toxic_unique.append(f"Chiral center configuration difference ({diff_type})")
        if ez_diff:
            toxic_unique.append(f"E/Z bond geometry difference ({diff_type})")

    return toxic_unique, nontoxic_unique
