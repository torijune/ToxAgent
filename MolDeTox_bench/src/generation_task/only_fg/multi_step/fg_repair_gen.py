"""Generate toxicity_repair_generation_multi QA set (only_fg / multi_step)"""
from typing import List, Dict, Any, Optional

from QA_templates.only_fg import OnlyFGQATemplate
from utils import (
    load_check_diff_csv, initialize_qa_components,
    extract_smiles, parse_fg_list,
    extract_dataset_info, extract_fg_uniques,
    save_qa_items_to_jsonl, create_base_context
,
    MolecularConverter)
from endpoint_desc import get_dataset_context



def generate_toxicity_repair_generation_multi_qa(
    csv_path: str,
    output_path: Optional[str] = None,
    smiles_format: str = "smiles",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate toxicity_repair_generation_multi QA items from only_fg_multi_step_diff.csv."""
    df = load_check_diff_csv(csv_path, diff_type='only_fg', step_num='multi_step')
    converter = MolecularConverter(smiles_format)
    template = OnlyFGQATemplate()
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

            tx_fgs = parse_fg_list(row.get('toxic_fg_names'))
            nt_fgs = parse_fg_list(row.get('nontoxic_fg_names'))

            tx_unique, nt_unique, atom_diff_fgs = extract_fg_uniques(row)

            # All pairs in check_diff have a diff. Use any available difference.
            if not tx_unique:
                tx_unique = atom_diff_fgs or nt_unique
            if not nt_unique:
                nt_unique = atom_diff_fgs or tx_unique
            if not tx_unique or not nt_unique:
                continue  # No FG difference (should not happen in check_diff data)

            if not tx_unique and atom_diff_fgs:
                tx_unique = atom_diff_fgs
            if not nt_unique and atom_diff_fgs:
                nt_unique = atom_diff_fgs
            if not tx_unique or not nt_unique:
                continue

            dataset_name, endpoint = extract_dataset_info(row)
            endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)

            question = template.create_question(
                nt_fgs=nt_fgs, tx_fgs=tx_fgs, nt_mol="", tx_mol=tx_mol,
                mol_format=mol_format, question_type="toxicity_repair_generation_multi",
                dataset_name=dataset_name, subset=endpoint
            ,
                step_type="multi_step"
            )
            if endpoint_desc:
                question = endpoint_desc + "\n\n" + question

            answer = template.create_answer(
                nt_unique=nt_unique, tx_unique=tx_unique,
                question_type="toxicity_repair_generation_multi",
                subset="TN", nt_mol=nt_mol
            ,
                step_type="multi_step"
            )

            qa_id = f"toxicity_repair_generation_multi_multi_step_{idx}"
            context = create_base_context(row, "toxicity_repair_generation_multi", tx_smiles, nt_smiles)
            context.update({"toxic_unique_fgs": tx_unique, "nontoxic_unique_fgs": nt_unique})
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
    qa_items = generate_toxicity_repair_generation_multi_qa(
        csv_path=args.csv_path, output_path=args.output_path,
        smiles_format=args.smiles_format, limit=args.limit
    )
    print(f"\nTotal QA items generated: {len(qa_items)}")
