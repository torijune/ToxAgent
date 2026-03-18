"""
Task3 instruction: remove only_toxic_safe_fragments, add only_nontoxic_safe_fragments instruction.

This is the canonical implementation for the task3_instruction naming.
"""

from typing import Optional


def build_cot_instruction(
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
    step: str = "multi_step",
) -> str:
    remove_s = (only_toxic_safe_fragments or "").strip()
    add_s = (only_nontoxic_safe_fragments or "").strip()

    if step == "single_step":
        remove_phrase = "Remove the following toxicity-associated fragment from the toxic molecule"
        add_phrase = "add the following non-toxic replacement fragment"
    else:
        remove_phrase = "Remove the following toxicity-associated fragment(s) from the toxic molecule"
        add_phrase = "add the following non-toxic replacement fragment(s)"

    parts = [
        f"Instruction (reasoning): {remove_phrase}: {remove_s!r}. "
        f"Then {add_phrase}: {add_s!r}. "
        "Output the resulting non-toxic molecule as a single SMILES string."
    ]
    return " ".join(parts).strip()

