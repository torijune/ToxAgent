"""Generate stereo_remove_add_repair_gen QA set (only_stereo / one_step)"""
from typing import List, Dict, Any, Optional
import sys

from QA_templates.only_stereo import OnlyStereoQATemplate
from utils import (
    load_check_diff_csv, initialize_qa_components,
    parse_stereo_dict, parse_fg_list, extract_smiles,
    extract_dataset_info, save_qa_items_to_jsonl, create_base_context
,
    MolecularConverter,
    explain_stereochemistry_for_repair)
from endpoint_desc import get_dataset_context


def _build_stereo_ez_safe(ez_raw: list) -> list:
    """Convert bond lists to tuples to avoid unhashable key errors in stereo_qa_builder."""
    ez = []
    for b in ez_raw:
        if isinstance(b, dict) and isinstance(b.get('bond'), list):
            b = dict(b)
            b['bond'] = tuple(b['bond'])
        ez.append(b)
    return ez


def _build_stereo_from_columns(row) -> dict:
    chiral = parse_fg_list(row.get('toxic_chiral_centers', '[]'))
    ez = _build_stereo_ez_safe(parse_fg_list(row.get('toxic_ez_bonds', '[]')))
    return {
        'chiral_centers': {'chiral_center_count': len(chiral), 'chiral_centers': chiral},
        'ez_bonds': {'ez_bond_count': len(ez), 'ez_bonds': ez},
        'has_stereochemistry': bool(chiral or ez)
    }


def _build_nontoxic_stereo_from_columns(row) -> dict:
    chiral = parse_fg_list(row.get('nontoxic_chiral_centers', '[]'))
    ez = _build_stereo_ez_safe(parse_fg_list(row.get('nontoxic_ez_bonds', '[]')))
    return {
        'chiral_centers': {'chiral_center_count': len(chiral), 'chiral_centers': chiral},
        'ez_bonds': {'ez_bond_count': len(ez), 'ez_bonds': ez},
        'has_stereochemistry': bool(chiral or ez)
    }


def generate_stereo_remove_add_repair_gen_qa(
    csv_path: str,
    output_path: Optional[str] = None,
    smiles_format: str = "smiles",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate stereo_remove_add_repair_gen QA items from only_stereo_one_step_diff.csv."""
    df = load_check_diff_csv(csv_path, diff_type='only_stereo', step_num='one_step')
    converter = MolecularConverter(smiles_format)
    template = OnlyStereoQATemplate()
    mol_format = "SELFIES" if smiles_format == "selfies" else "SMILES"

    qa_items = []
    for idx, row in df.iterrows():
        if limit and len(qa_items) >= limit:
            break
        try:
            tx_smiles, nt_smiles = extract_smiles(row)
            if not tx_smiles or not nt_smiles:
                continue

            tx_mol = converter.convert(tx_smiles)
            nt_mol = converter.convert(nt_smiles)

            toxic_stereo = parse_stereo_dict(row.get('toxic_stereochemistry', {}))
            nontoxic_stereo = parse_stereo_dict(row.get('nontoxic_stereochemistry', {}))
            stereo_diff = parse_stereo_dict(row.get('stereochemistry_difference', {}))

            if not toxic_stereo:
                toxic_stereo = _build_stereo_from_columns(row)
            if not nontoxic_stereo:
                nontoxic_stereo = _build_nontoxic_stereo_from_columns(row)

            if not toxic_stereo.get('has_stereochemistry') and not nontoxic_stereo.get('has_stereochemistry'):
                continue

            comparison = {
                'toxic_stereochemistry': toxic_stereo,
                'nontoxic_stereochemistry': nontoxic_stereo,
                'stereochemistry_difference': stereo_diff
            }
            try:
                stereo_summary = explain_stereochemistry_for_repair(comparison, add_atom_indices=True)
            except Exception:
                stereo_summary = "Stereochemical differences exist between the molecules."

            dataset_name, endpoint = extract_dataset_info(row)
            endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)

            question = template._create_stereo_remove_add_repair_question(
                tx_mol=tx_mol, nt_mol=nt_mol, mol_format=mol_format,
                stereo_explanation=stereo_summary,
                dataset_name=dataset_name, subset=endpoint,
                use_atom_numbering=(smiles_format == "smiles"),
                step_type="one_step"
            )
            if endpoint_desc:
                question = endpoint_desc + "\n\n" + question

            answer = template._create_stereo_remove_add_repair_answer(nt_mol)

            qa_id = f"stereo_remove_add_repair_gen_one_step_{idx}"
            context = create_base_context(row, "stereo_remove_add_repair_gen", tx_smiles, nt_smiles)
            context.update({
                "toxic_stereochemistry": toxic_stereo,
                "nontoxic_stereochemistry": nontoxic_stereo,
                "stereochemistry_difference": stereo_diff
            })
            qa_items.append({"id": qa_id, "question": question, "answer": answer, "context": context})

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    print(f"Generated {len(qa_items)} QA items")
    if output_path:
        save_qa_items_to_jsonl(qa_items, output_path)
    return qa_items


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--smiles-format", type=str, default="smiles", choices=["smiles", "selfies"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    qa_items = generate_stereo_remove_add_repair_gen_qa(
        csv_path=args.csv_path, output_path=args.output_path,
        smiles_format=args.smiles_format, limit=args.limit
    )
    print(f"\nTotal QA items generated: {len(qa_items)}")
