"""Generate stereo_identification QA set (only_stereo / one_step)"""
from typing import List, Dict, Any, Optional

from QA_templates.only_stereo import OnlyStereoQATemplate
from utils import (
    load_check_diff_csv, initialize_qa_components,
    extract_smiles, extract_dataset_info,
    extract_stereo_uniques, build_stereo_uniques_from_columns,
    save_qa_items_to_jsonl, create_base_context
,
    MolecularConverter)
from endpoint_desc import get_dataset_context



def generate_stereo_identification_qa(
    csv_path: str,
    output_path: Optional[str] = None,
    smiles_format: str = "smiles",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate stereo_identification QA items from only_stereo_one_step_diff.csv."""
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

            # Read pre-extracted unique stereo directly from columns
            toxic_unique_stereo, nontoxic_unique_stereo = extract_stereo_uniques(row)

            # Fallback: build from individual chiral/ez columns if pre-extracted is empty
            if not toxic_unique_stereo:
                toxic_unique_stereo, nontoxic_unique_stereo = build_stereo_uniques_from_columns(row)
            # If toxic has no unique stereo but nontoxic does, the absence is the key difference
            if not toxic_unique_stereo and nontoxic_unique_stereo:
                toxic_unique_stereo = [f"Lacks: {s}" for s in nontoxic_unique_stereo]
            if not toxic_unique_stereo:
                continue  # No stereo difference found (should not happen in check_diff data)

            dataset_name, endpoint = extract_dataset_info(row)
            endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)

            question = template._create_stereo_identification_question(
                tx_mol=tx_mol, nt_mol=nt_mol, mol_format=mol_format,
                dataset_name=dataset_name, subset=endpoint
            ,
                step_type="one_step"
            )
            if endpoint_desc:
                question = endpoint_desc + "\n\n" + question

            answer = template._create_stereo_identification_answer(toxic_unique_stereo)

            qa_id = f"stereo_identification_one_step_{idx}"
            context = create_base_context(row, "stereo_identification", tx_smiles, nt_smiles)
            context.update({"toxic_unique_stereo": toxic_unique_stereo})
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
    qa_items = generate_stereo_identification_qa(
        csv_path=args.csv_path, output_path=args.output_path,
        smiles_format=args.smiles_format, limit=args.limit
    )
    print(f"\nTotal QA items generated: {len(qa_items)}")
