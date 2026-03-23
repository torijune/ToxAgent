'''
Raw data: pairs_safe_filtered_valid.csv (or equivalent split CSV)
Columns: toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles, toxic_safe, nontoxic_safe, only_toxic_safe_fragments, only_nontoxic_safe_fragments, dataset_name, endpoint

Tasks:
  subtask1_safe_to_smiles             : SAFE string -> SMILES
  subtask2_smiles_to_safe             : SMILES -> SAFE string
  task1_toxic_fragment_identification : toxic SAFE -> only_toxic_safe_fragments
  task2_nontoxic_fragment_generation  : toxic SAFE + only_toxic_frags -> only_nontoxic_safe_fragments
  task3_nontoxic_smiles_generation    : toxic SAFE -> nontoxic SMILES (end-to-end)
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
    """Context for Task 1, 2, 3: pairs are structurally similar, same endpoint, only toxicity differs."""
    return (
        "Context: The toxic and non-toxic molecules in this task form a pair that is structurally very similar "
        "with minimal physicochemical difference; they differ only in toxicity versus non-toxicity for the same endpoint. "
        "Keep this in mind when performing the task.\n\n"
    )


def _preserve_properties_instruction() -> str:
    """Instruction for Task 2 and 3: preserve other properties, only reduce toxicity for the endpoint."""
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
    task_name: str,
    toxic_safe: str,
    nontoxic_safe: str,
    toxic_safe_decoded_smiles: str,
    nontoxic_safe_decoded_smiles: str,
    molecule_repr: str = "both_repre",
) -> str:
    """
    Task 2 (nontoxic_fragment_generation), Task 1 (toxic_fragment_identification),
    Task 3 (nontoxic_smiles_generation)용:
    question에 molecule representation을 주기 위한 블록.
    Subtask 1/2는 제외.

    molecule_repr: "only_smiles" | "only_safe" | "both_repre"
      - only_smiles: SMILES만 표시
      - only_safe: SAFE만 표시
      - both_repre: SMILES와 SAFE 둘 다 표시 + 동일 molecule임을 명시
    """
    t_safe = (toxic_safe or "").strip()
    n_safe = (nontoxic_safe or "").strip()
    t_smiles = (toxic_safe_decoded_smiles or "").strip()
    n_smiles = (nontoxic_safe_decoded_smiles or "").strip()
    repr_type = (molecule_repr or "both_repre").strip().lower()

    if task_name in ("subtask1", "subtask2"):
        return ""

    def _toxic_line() -> str:
        if repr_type == "only_smiles":
            return f"- Toxic molecule: SMILES = {t_smiles!r}" if t_smiles else ""
        if repr_type == "only_safe":
            return f"- Toxic molecule: SAFE = {t_safe!r}" if t_safe else ""
        # both_repre
        if t_smiles or t_safe:
            return f"- Toxic molecule (same molecule): SMILES = {t_smiles!r}, SAFE = {t_safe!r}"
        return ""

    if task_name in ("task1", "task2", "task3"):
        lt = _toxic_line()
        if not lt:
            return ""
        # strip "- Toxic molecule: " or "- Toxic molecule (same molecule): " prefix
        content = lt.replace("- Toxic molecule (same molecule): ", "").replace("- Toxic molecule: ", "").strip()
        return "Full molecule representation (toxic): " + content + "\n\n"
    return ""


def toxic_molecule_content_for_repr(
    toxic_safe: str,
    toxic_safe_decoded_smiles: str,
    molecule_repr: str = "both_repre",
) -> str:
    """
    `_smiles_safe_matching` (task1/2/3)과 동일한 규칙으로 toxic molecule을 한 줄로 표현한다.
    ICL few-shot 예시에서 본문의 molecule representation과 맞출 때 사용한다.

    molecule_repr: "only_smiles" | "only_safe" | "both_repre"
    반환 예: ``SMILES = '...'``, ``SAFE = '...'``, 또는 ``SMILES = '...', SAFE = '...'``.
    """
    t_safe = (toxic_safe or "").strip()
    t_smiles = (toxic_safe_decoded_smiles or "").strip()
    repr_type = (molecule_repr or "both_repre").strip().lower()
    if repr_type == "only_smiles":
        return f"SMILES = {t_smiles!r}" if t_smiles else ""
    if repr_type == "only_safe":
        return f"SAFE = {t_safe!r}" if t_safe else ""
    if t_smiles or t_safe:
        return f"SMILES = {t_smiles!r}, SAFE = {t_safe!r}"
    return ""


def task2_nontoxic_fragment_generation(
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
    molecule_repr: str = "both_repre",
) -> tuple:
    """Task 2: nontoxic_fragment_generation — generate question and answer in English.

    Given the toxic molecule's SAFE and its toxicity-associated fragment(s), output the
    replacement nontoxic fragment(s).

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
        "task2", toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles,
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task2_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_nontoxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task2_fragment_line = (
            f"- The single fragment that appears only in the toxic molecule (candidate for toxicity-associated structure for this endpoint) is: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragment—i.e. the single SAFE fragment that, when used in place of the only_toxic_safe_fragment, yields a non-toxic molecule for this endpoint. "
            + _preserve_properties_instruction()
        )
    else:
        task2_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_nontoxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task2_fragment_line = (
            f"- The fragments that appear only in the toxic molecule (candidates for toxicity-associated structure for this endpoint) are: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragments—i.e. the SAFE fragment(s) that, when used in place of the only_toxic_safe_fragments, yield a non-toxic molecule for this endpoint. "
            + _preserve_properties_instruction()
        )

    task2_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task2_fragment_line
    ).strip()
    if include_output_format:
        task2_question += " " + task2_output_format

    task2_answer = {"answer": only_nontoxic_safe_fragments or ""}

    return task2_question, task2_answer


def subtask2_smiles_to_safe(
    smiles: str,
    safe: str,
    include_output_format: bool = True,
) -> tuple:
    """Subtask 2: smiles_to_safe — generate question and answer in English.

    Given a SMILES string, output its SAFE representation.
    """
    safe_explanation = _build_safe_explanation()
    task2_output_format = (
        'Output format: a single JSON object with key "answer" and value the SAFE string '
        '(dot-separated if multiple). Example: {"answer": "frag1.frag2"}'
    )

    smiles = (smiles or "").strip()
    safe_str = (safe or "").strip()

    parts = [safe_explanation, ""]
    parts.append(f"- Original SMILES: {smiles}")
    parts.append(
        "\nTask: Convert this molecule into its SAFE representation string (dot-separated fragments) "
        "and output that string exactly."
    )
    if include_output_format:
        parts.append(" " + task2_output_format)

    task2_question = "\n".join(parts).strip()
    task2_answer = {"answer": safe_str}
    return task2_question, task2_answer


def subtask1_safe_to_smiles(
    safe: str,
    smiles: str,
    include_output_format: bool = True,
) -> tuple:
    """Subtask 1: safe_to_smiles — given SAFE string, generate question and answer (output SMILES)."""
    safe_explanation = _build_safe_explanation()
    task_output_format = (
        'Output format: a single JSON object with key "answer" and value the SMILES string. '
        'Example: {"answer": "CCO"}'
    )

    safe_str = (safe or "").strip()
    smiles_str = (smiles or "").strip()

    parts = [safe_explanation, ""]
    parts.append(f"- SAFE representation (dot-separated fragments): {safe_str}")
    parts.append(
        "\nTask: Reconstruct the full molecule from this SAFE representation and output its SMILES string."
    )
    if include_output_format:
        parts.append(" " + task_output_format)

    question = "\n".join(parts).strip()
    answer = {"answer": smiles_str}
    return question, answer


def task1_toxic_fragment_identification(
    toxic_safe: str,
    only_toxic_safe_fragments: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """
    Task 1: toxic_fragment_identification — generate question and answer in English.

    Given a toxic molecule's SAFE string, identify the toxicity-associated fragment(s).

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
        "task1", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    toxic_safe = (toxic_safe or "").strip()

    if step == "single_step":
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_toxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task1_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the single fragment that is the candidate for toxicity-associated structure (the part that drives toxicity for this endpoint) "
            "and output it as only_toxic_safe_fragment. "
        )
    else:
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_toxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task1_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the fragment(s) that are candidates for toxicity-associated structure (the part(s) that drive toxicity for this endpoint) "
            "and output them as only_toxic_safe_fragments (dot-separated if multiple). "
        )

    task1_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + f"- Toxic molecule (SAFE representation): {toxic_safe}\n\n"
        + task1_instruction
    ).strip()
    if include_output_format:
        task1_question += " " + task1_output_format

    task1_answer = {"answer": (only_toxic_safe_fragments or "").strip()}
    return task1_question, task1_answer


def task3_nontoxic_smiles_generation(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """Task 3: nontoxic_smiles_generation — end-to-end.

    LLM receives toxic molecule's SAFE and SMILES; performs identification (Task 1) and
    replacement (Task 2) in one step; outputs nontoxic_safe_decoded_smiles (full SMILES of
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
        "task3", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task3_instruction = (
            "Task: From the toxic molecule above, identify the single fragment that is the candidate for "
            "toxicity-associated structure for this endpoint, then determine the single replacement fragment "
            "that yields a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            + _preserve_properties_instruction()
        )
    else:
        task3_instruction = (
            "Task: From the toxic molecule above, identify the fragment(s) that are candidates for "
            "toxicity-associated structure for this endpoint, then determine the replacement fragment(s) "
            "that yield a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            + _preserve_properties_instruction()
        )

    task3_output_format = (
        'Output format: a single JSON object with key "answer" and value the nontoxic_safe_decoded_smiles string. '
        'Example: {"answer": "CCO"}'
    )

    task3_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task3_instruction
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (nontoxic_safe_decoded_smiles or "").strip()}
    return task3_question, task3_answer


def task3_instruction_nontoxic_smiles_generation(
    toxic_safe: str,
    cot_instruction: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """Task 3 instruction: nontoxic_smiles_generation with explicit remove/add instruction.

    Same as task3_nontoxic_smiles_generation (toxic SAFE -> nontoxic SMILES), but the prompt
    includes a CoT block that explicitly says: remove only_toxic_safe_fragments, add
    only_nontoxic_safe_fragments, then output the resulting SMILES.
    cot_instruction is built from only_toxic_safe_fragments and only_nontoxic_safe_fragments
    (e.g. via task3_instruction_ver.build_cot_instruction) and is typically loaded from
    merged_train.csv or merged_test.csv per row.
    """
    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    if endpoint_desc:
        endpoint_block = endpoint_desc.strip() + "\n\n"
    else:
        endpoint_block = ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        "task3", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    cot_block = (cot_instruction or "").strip()
    if cot_block:
        cot_block = cot_block + "\n\n"

    task3_output_format = (
        'Output format: a single JSON object with key "answer" and value the nontoxic_safe_decoded_smiles string. '
        'Example: {"answer": "CCO"}'
    )

    task3_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + cot_block
        + "Task: Apply the instruction above and output the resulting non-toxic molecule as a single SMILES string."
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (nontoxic_safe_decoded_smiles or "").strip()}
    return task3_question, task3_answer


def task3_stepwise_cot_nontoxic_smiles_generation(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    # Gold labels for evaluation (NOT shown in the question)
    only_toxic_safe_fragments: str = "",
    only_nontoxic_safe_fragments: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """
    Task 3 (new CoT version): one-call, step-by-step reasoning with structured outputs.

    The model must perform:
      Step 1) Toxic fragment identification (like Task1) -> only_toxic_safe_fragments
      Step 2) Nontoxic fragment generation (like Task2)  -> only_nontoxic_safe_fragments
      Step 3) Apply remove/add to produce final nontoxic SMILES (like Task3) -> answer

    Output is a single JSON object containing both intermediate step outputs and the final answer,
    so that we can parse and evaluate step correctness as well as final SMILES correctness.

    Notes:
    - only_toxic_safe_fragments / only_nontoxic_safe_fragments parameters are gold labels and are NOT
      included in the question text.
    """
    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    endpoint_block = (endpoint_desc.strip() + "\n\n") if endpoint_desc else ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        "task3",
        (toxic_safe or "").strip(),
        "",
        (toxic_safe_decoded_smiles or "").strip(),
        "",
        molecule_repr=molecule_repr,
    )

    # IMPORTANT: To be compatible with strict JSON schema parsing, keep output keys fixed
    # regardless of single_step/multi_step. Use dot-separated strings; single_step is just 1 token.
    step1_name = "only_toxic_safe_fragments"
    step2_name = "only_nontoxic_safe_fragments"
    if step == "single_step":
        step1_hint = "Identify the single fragment most likely responsible for toxicity for this endpoint."
        step2_hint = (
            "Propose a single non-toxic replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )
    else:
        step1_hint = (
            "Identify the fragment(s) most likely responsible for toxicity for this endpoint (dot-separated if multiple)."
        )
        step2_hint = (
            "Propose non-toxic replacement fragment(s) (dot-separated if multiple) that reduce toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )

    output_format = (
        "Output format: a single JSON object with the following keys:\n"
        f'- "step1_{step1_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step1_reasoning": string\n'
        f'- "step2_{step2_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step2_reasoning": string\n'
        '- "step3_reasoning": string\n'
        '- "answer": string (the final nontoxic SMILES)\n'
        'Example: {"step1_only_toxic_safe_fragments":"frag1.frag2","step1_reasoning":"...","step2_only_nontoxic_safe_fragments":"fragA.fragB","step2_reasoning":"...","step3_reasoning":"...","answer":"CCO"}'
    )

    task_block = (
        "Task: Solve the following in ONE call, step by step, using natural-language reasoning.\n"
        "\n"
        "Step 1 (endpoint-aware toxic fragment identification):\n"
        f"- {step1_hint}\n"
        "- In step1_reasoning, identify which fragment is most likely responsible for toxicity for this endpoint and explain *why* the fragment(s) are toxicity-associated for this endpoint, using brief chemical intuition (no need for citations).\n"
        f"- Output the fragment string as step1_{step1_name}.\n"
        "\n"
        "Step 2 (endpoint-aware non-toxic fragment proposal):\n"
        "- Using the Step 1 fragment as the part to be replaced, propose replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        f"- {step2_hint}\n"
        "- In step2_reasoning, explain the design intent: what property/alert you are trying to reduce for this endpoint and what you preserve while keeping the overall scaffold as similar as possible.\n"
        f"- Output the fragment string as step2_{step2_name}.\n"
        "\n"
        "Step 3 (construct final non-toxic SMILES):\n"
        "- Combine Step 1 and Step 2: conceptually remove the toxic fragment and add the proposed non-toxic fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        "- In step3_reasoning, describe at a high level how the final molecule changes relative to the toxic molecule.\n"
        '- Output the final non-toxic molecule as a single SMILES string under the key "answer".\n'
        "\n"
        "Important:\n"
        f"- {_preserve_properties_instruction()}\n"
        "- Your output must be a SINGLE JSON object.\n"
        "- Do not output any text outside the JSON.\n"
        "- The fragment fields must be SAFE fragment strings (dot-separated if multiple)."
    )

    question = (
        endpoint_block
        + safe_explanation
        + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task_block
    ).strip()
    if include_output_format:
        question += "\n\n" + output_format

    answer = {
        "answer": (nontoxic_safe_decoded_smiles or "").strip(),
        "gold_only_toxic_safe_fragments": (only_toxic_safe_fragments or "").strip(),
        "gold_only_nontoxic_safe_fragments": (only_nontoxic_safe_fragments or "").strip(),
    }
    return question, answer


def task3_stepwise_cot_nontoxic_safe_generation(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe: str = "",
    # Gold labels for evaluation (NOT shown in the question)
    only_toxic_safe_fragments: str = "",
    only_nontoxic_safe_fragments: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """
    Task 3 stepwise CoT 변형: Step1/2는 SMILES 버전과 동일(SAFE fragment), 최종 출력만 **전체 nontoxic SAFE 문자열**.

    Gold `answer`는 `nontoxic_safe`(full SAFE)이며, 평가는 task3_nontoxic_safe_generation과 동일한 SAFE/SMILES 메트릭을
    Step3 최종 출력에 적용한다.
    """
    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    endpoint_block = (endpoint_desc.strip() + "\n\n") if endpoint_desc else ""

    safe_explanation = _build_safe_explanation()
    pair_context = _pair_context_for_toxic_nontoxic_tasks()
    full_mol_block = _smiles_safe_matching(
        "task3",
        (toxic_safe or "").strip(),
        "",
        (toxic_safe_decoded_smiles or "").strip(),
        "",
        molecule_repr=molecule_repr,
    )

    step1_name = "only_toxic_safe_fragments"
    step2_name = "only_nontoxic_safe_fragments"
    if step == "single_step":
        step1_hint = "Identify the single fragment most likely responsible for toxicity for this endpoint."
        step2_hint = (
            "Propose a single non-toxic replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )
    else:
        step1_hint = (
            "Identify the fragment(s) most likely responsible for toxicity for this endpoint (dot-separated if multiple)."
        )
        step2_hint = (
            "Propose non-toxic replacement fragment(s) (dot-separated if multiple) that reduce toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )

    output_format = (
        "Output format: a single JSON object with the following keys:\n"
        f'- "step1_{step1_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step1_reasoning": string\n'
        f'- "step2_{step2_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step2_reasoning": string\n'
        '- "step3_reasoning": string\n'
        '- "answer": string (the final nontoxic **full SAFE string** for the whole molecule)\n'
        'Example: {"step1_only_toxic_safe_fragments":"frag1.frag2","step1_reasoning":"...","step2_only_nontoxic_safe_fragments":"fragA.fragB","step2_reasoning":"...","step3_reasoning":"...","answer":"CCO.[*:1]"}'
    )

    task_block = (
        "Task: Solve the following in ONE call, step by step, using natural-language reasoning.\n"
        "\n"
        "Step 1 (endpoint-aware toxic fragment identification):\n"
        f"- {step1_hint}\n"
        "- In step1_reasoning, identify which fragment is most likely responsible for toxicity for this endpoint and explain *why* the fragment(s) are toxicity-associated for this endpoint, using brief chemical intuition (no need for citations).\n"
        f"- Output the fragment string as step1_{step1_name}.\n"
        "\n"
        "Step 2 (endpoint-aware non-toxic fragment proposal):\n"
        "- Using the Step 1 fragment as the part to be replaced, propose replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        f"- {step2_hint}\n"
        "- In step2_reasoning, explain the design intent: what property/alert you are trying to reduce for this endpoint and what you preserve while keeping the overall scaffold as similar as possible.\n"
        f"- Output the fragment string as step2_{step2_name}.\n"
        "\n"
        "Step 3 (construct final non-toxic SAFE):\n"
        "- Combine Step 1 and Step 2: conceptually remove the toxic fragment and add the proposed non-toxic fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        "- In step3_reasoning, describe at a high level how the final molecule changes relative to the toxic molecule.\n"
        '- Output the final non-toxic **full molecule SAFE string** under the key "answer" (not SMILES).\n'
        "\n"
        "Important:\n"
        f"- {_preserve_properties_instruction()}\n"
        "- Your output must be a SINGLE JSON object.\n"
        "- Do not output any text outside the JSON.\n"
        "- The fragment fields must be SAFE fragment strings (dot-separated if multiple)."
    )

    question = (
        endpoint_block
        + safe_explanation
        + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task_block
    ).strip()
    if include_output_format:
        question += "\n\n" + output_format

    answer = {
        "answer": (nontoxic_safe or "").strip(),
        "gold_only_toxic_safe_fragments": (only_toxic_safe_fragments or "").strip(),
        "gold_only_nontoxic_safe_fragments": (only_nontoxic_safe_fragments or "").strip(),
    }
    return question, answer


def task3_nontoxic_safe_generation(
    toxic_safe: str,
    nontoxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
) -> tuple:
    """Task 3: nontoxic_safe_generation — end-to-end.

    LLM receives toxic molecule's SAFE and SMILES; performs identification (Task 1) and
    replacement (Task 2) in one step; outputs nontoxic_safe (full SAFE string of
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
        "task3", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task3_instruction = (
            "Task: From the toxic molecule above, identify the single fragment that is the candidate for "
            "toxicity-associated structure for this endpoint, then determine the single replacement fragment "
            "that yields a non-toxic molecule. Output the resulting non-toxic molecule as a single SAFE string "
            "as the nontoxic SAFE string. "
            + _preserve_properties_instruction()
        )
    else:
        task3_instruction = (
            "Task: From the toxic molecule above, identify the fragment(s) that are candidates for "
            "toxicity-associated structure for this endpoint, then determine the replacement fragment(s) "
            "that yield a non-toxic molecule. Output the resulting non-toxic molecule as a single SAFE string "
            "as the nontoxic SAFE string. "
            + _preserve_properties_instruction()
        )

    task3_output_format = (
        'Output format: a single JSON object with key "answer" and value the resulting non-toxic SAFE string. '
        'Example: {"answer": "CCO.[*:1]"}'
    )

    task3_question = (
        endpoint_block
        + safe_explanation + "\n\n"
        + pair_context
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task3_instruction
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (nontoxic_safe or "").strip()}
    return task3_question, task3_answer