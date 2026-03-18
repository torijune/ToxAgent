'''
Raw data: commom_frage_pairs_with_smiles.csv
Columns: toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles, toxic_safe, nontoxic_safe, only_toxic_safe_fragments, only_nontoxic_safe_fragments, dataset_name, endpoint

# Task 1: toxic_safe_to_nontoxic_safe
- question: [Endpoint Description] + [SAFE string explanation] + task instruction (all in English)
- answer: only_nontoxic_safe_fragments
'''

import sys
from pathlib import Path
from typing import Optional

# MolDeTox_bench/src/endpoint_desc에서 Endpoint Description 가져오기
_QA_SRC = Path(__file__).resolve().parent
_PROJECT_ROOT = _QA_SRC.parent.parent.parent
_MOLDETOX_SRC = _PROJECT_ROOT / "MolDeTox_bench" / "src"
if str(_MOLDETOX_SRC) not in sys.path:
    sys.path.insert(0, str(_MOLDETOX_SRC))

try:
    from endpoint_desc import get_dataset_context
except ImportError:
    get_dataset_context = lambda dataset_name=None, endpoint=None: ""


def _pair_context_for_toxic_nontoxic_tasks() -> str:
    """Context for Task 1, 3, 4: pairs are structurally similar, same endpoint, only toxicity differs."""
    return (
        "Context: The toxic and non-toxic molecules in this task form a pair that is structurally very similar "
        "with minimal physicochemical difference; they differ only in toxicity versus non-toxicity for the same endpoint. "
        "Keep this in mind when performing the task.\n\n"
    )


def _preserve_properties_instruction() -> str:
    """Instruction for Task 1 and 4: preserve other properties, only reduce toxicity for the endpoint."""
    return (
        "When modifying the toxic molecule to make it non-toxic, do not change other physicochemical or "
        "pharmacological properties; only reduce or remove the drug toxicity for this endpoint. "
    )


def _build_safe_explanation() -> str:
    """Return a concise generic SAFE string explanation."""
    return (
        "SAFE (Sequential Attachment-based Fragment Embedding) is a SMILES-compatible string representation "
        "that expresses a molecule as a dot-separated sequence of fragments.\n"
        "\n"
        "How SAFE is constructed:\n"
        "- **Fragmentation**: A molecule is split into fragments by cutting selected bonds using a slicing algorithm.\n"
        "- **Slicer**: The default slicer is `brics`, a rule-based method that cuts retrosynthetically relevant bonds "
        "to produce chemically meaningful substructures.\n"
        "- **Attachment Markers**: At each cut site, attachment information is encoded with SMILES-style ring-closure digits "
        "(e.g., `1`, `2`, ..., `%10`). Matching digits across fragments indicate where fragments reconnect in the full molecule.\n"
        "- **Serialization**: The resulting fragments are written as SMILES strings and joined with `.` separators to form a SAFE string.\n"
        "\n"
        "Important characteristics:\n"
        "- **Fragment-based representation**: Each token block corresponds to a substructure rather than the entire molecule.\n"
        "- **Order invariance**: Changing the fragment order does not change the reconstructed molecule.\n"
        "- **Partial structures**: Individual fragments may look chemically incomplete on their own because they are parts of a larger graph."
    )

def _smiles_safe_matching(
    task_number: int,
    toxic_safe: str,
    nontoxic_safe: str,
    toxic_safe_decoded_smiles: str,
    nontoxic_safe_decoded_smiles: str,
) -> str:
    """
    Task 1, 3용: question에 full molecule representation (SMILES + SAFE)을 함께 주기 위한 블록.
    Task 2는 제외.
    """
    t_safe = (toxic_safe or "").strip()
    n_safe = (nontoxic_safe or "").strip()
    t_smiles = (toxic_safe_decoded_smiles or "").strip()
    n_smiles = (nontoxic_safe_decoded_smiles or "").strip()

    if task_number == 2:
        return ""
    if task_number == 1:
        lines = ["Full molecule representations:"]
        lines.append(f"- Toxic molecule: SMILES = {t_smiles!r}, SAFE = {t_safe!r}")
        if n_smiles or n_safe:
            lines.append(f"- Nontoxic molecule: SMILES = {n_smiles!r}, SAFE = {n_safe!r}")
        return "\n".join(lines) + "\n\n" if len(lines) > 1 else ""
    if task_number == 3 or task_number == 4:
        if not (t_smiles or t_safe):
            return ""
        return f"Full molecule representation (toxic): SMILES = {t_smiles!r}, SAFE = {t_safe!r}\n\n"
    return ""


def task1_toxic_safe_to_nontoxic_safe(
    toxic_safe: str,
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    nontoxic_safe: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
) -> tuple:
    """Task 1: toxic_safe_to_nontoxic_safe — generate question and answer in English.

    step: "single_step" (one fragment) or "multi_step" (multiple fragments). Affects question wording and output format.
    include_output_format: if False, question ends without open-ended output format (for MCQA).
    """
    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    if endpoint_desc:
        endpoint_block = endpoint_desc.strip() + "\n\n"
    else:
        endpoint_block = ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        1, toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles
    )

    if step == "single_step":
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_nontoxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task1_fragment_line = (
            f"- The single fragment that appears only in the toxic molecule (candidate for toxicity-associated structure for this endpoint) is: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragment—i.e. the single SAFE fragment that, when used in place of the only_toxic_safe_fragment, yields a non-toxic molecule for this endpoint. "
            # + _preserve_properties_instruction()
        )
    else:
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_nontoxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task1_fragment_line = (
            f"- The fragments that appear only in the toxic molecule (candidates for toxicity-associated structure for this endpoint) are: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragments—i.e. the SAFE fragment(s) that, when used in place of the only_toxic_safe_fragments, yield a non-toxic molecule for this endpoint. "
            # + _preserve_properties_instruction()
        )

    task1_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task1_fragment_line
    ).strip()
    if include_output_format:
        task1_question += " " + task1_output_format

    task1_answer = {"answer": only_nontoxic_safe_fragments or ""}

    return task1_question, task1_answer


def task2_smiles_to_safe(
    smiles: str,
    safe: str,
    include_output_format: bool = True,
) -> tuple:
    """Task 2: smiles_to_safe — generate question and answer in English.

    include_output_format: if False, question ends without open-ended output format (for MCQA).
    """
    safe_explanation = _build_safe_explanation()
    task2_output_format = (
        'Output format: a single JSON object with key "answer" and value the SAFE string '
        '(dot-separated if multiple). Example: {"answer": "frag1.frag2"}'
    )

    smiles = (smiles or "").strip()

    parts = [safe_explanation, ""]
    parts.append(f"- Original SMILES: {smiles}")
    parts.append(
        "\nTask: Convert this molecule into its SAFE representation string (dot-separated fragments) "
        "and output that string exactly."
    )
    if include_output_format:
        parts.append(" " + task2_output_format)

    task2_question = "\n".join(parts).strip()
    task2_answer = {"answer": (safe or "").strip()}
    return task2_question, task2_answer


def task3_toxic_fragment_identification(
    toxic_safe: str,
    only_toxic_safe_fragments: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
) -> tuple:
    """
    Task 3: toxic_fragment_identification — generate question and answer in English.

    step: "single_step" (one fragment) or "multi_step" (multiple fragments). Affects question wording and output format.
    include_output_format: if False, question ends without open-ended output format (for MCQA).
    """

    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    if endpoint_desc:
        endpoint_block = endpoint_desc.strip() + "\n\n"
    else:
        endpoint_block = ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        3, (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), ""
    )

    toxic_safe = (toxic_safe or "").strip()

    if step == "single_step":
        task3_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_toxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task3_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the single fragment that is the candidate for toxicity-associated structure (the part that drives toxicity for this endpoint) "
            "and output it as only_toxic_safe_fragment. "
        )
    else:
        task3_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_toxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task3_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the fragment(s) that are candidates for toxicity-associated structure (the part(s) that drive toxicity for this endpoint) "
            "and output them as only_toxic_safe_fragments (dot-separated if multiple). "
        )

    task3_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + f"- Toxic molecule (SAFE representation): {toxic_safe}\n\n"
        + task3_instruction
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (only_toxic_safe_fragments or "").strip()}
    return task3_question, task3_answer


def task4_safe_to_nontoxic_smiles(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
) -> tuple:
    """Task 4: safe_to_nontoxic_smiles — end-to-end.

    LLM receives toxic molecule's SAFE and SMILES; performs identification (Task 3) and
    replacement (Task 1) in one step; outputs nontoxic_safe_decoded_smiles (full SMILES of
    the non-toxic molecule) as the answer.
    """
    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    if endpoint_desc:
        endpoint_block = endpoint_desc.strip() + "\n\n"
    else:
        endpoint_block = ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        4, (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), ""
    )

    if step == "single_step":
        task4_instruction = (
            "Task: From the toxic molecule above, identify the single fragment that is the candidate for "
            "toxicity-associated structure for this endpoint, then determine the single replacement fragment "
            "that yields a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            # + _preserve_properties_instruction()
        )
    else:
        task4_instruction = (
            "Task: From the toxic molecule above, identify the fragment(s) that are candidates for "
            "toxicity-associated structure for this endpoint, then determine the replacement fragment(s) "
            "that yield a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            # + _preserve_properties_instruction()
        )

    task4_output_format = (
        'Output format: a single JSON object with key "answer" and value the nontoxic_safe_decoded_smiles string. '
        'Example: {"answer": "CCO"}'
    )

    task4_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task4_instruction
    ).strip()
    if include_output_format:
        task4_question += " " + task4_output_format

    task4_answer = {"answer": (nontoxic_safe_decoded_smiles or "").strip()}
    return task4_question, task4_answer