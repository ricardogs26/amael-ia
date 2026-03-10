"""
Groups a flat plan list into batches for parallel execution.

Rule:
  - Consecutive non-REASONING steps → one parallel batch
  - REASONING steps → always alone (they synthesise accumulated context)

Examples:
  ["K8S_TOOL: A", "RAG_RETRIEVAL: B", "REASONING: C"]
  → [["K8S_TOOL: A", "RAG_RETRIEVAL: B"], ["REASONING: C"]]

  ["K8S_TOOL: A", "REASONING: B", "K8S_TOOL: C", "REASONING: D"]
  → [["K8S_TOOL: A"], ["REASONING: B"], ["K8S_TOOL: C"], ["REASONING: D"]]
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


def group_plan_into_batches(plan: List[str]) -> List[List[str]]:
    batches: List[List[str]] = []
    tool_batch: List[str] = []

    for step in plan:
        if step.upper().startswith("REASONING:"):
            if tool_batch:
                batches.append(tool_batch)
                tool_batch = []
            batches.append([step])
        else:
            tool_batch.append(step)

    if tool_batch:
        batches.append(tool_batch)

    parallel_steps = sum(len(b) for b in batches if len(b) > 1)
    if parallel_steps:
        logger.info(
            f"[GROUPER] {len(batches)} grupos | {parallel_steps}/{len(plan)} pasos en paralelo"
        )
    else:
        logger.info(f"[GROUPER] {len(batches)} grupos (todos secuenciales)")

    return batches
