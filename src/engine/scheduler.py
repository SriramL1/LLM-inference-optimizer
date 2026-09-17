"""
Stage 5: continuous batching scheduler.

Drives ContinuousBatchingEngine with dynamic admission: as soon as a
sequence finishes (reaches its max_new_tokens), its slot is freed and
the next pending request is admitted immediately -- rather than static
batching, where the whole batch has to finish before any new request
starts. This is the actual mechanism that improves real-world throughput
in a serving system, since request lengths vary and static batches waste
GPU time waiting on the longest sequence in the batch.

Scope: fixed max_new_tokens per request, no EOS-based early stopping, no
priority/fairness policy beyond simple FCFS admission. A real scheduler
(vLLM's) also handles preemption when the page pool is exhausted; this
one assumes max_batch_size and max_pages are sized to avoid that case,
consistent with Stage 4's own stated scope (no eviction policy).
"""
from typing import Dict, List, Tuple

import torch

from src.engine.continuous_batching_engine import ContinuousBatchingEngine


class ContinuousBatchingScheduler:
    def __init__(self, engine: ContinuousBatchingEngine, max_batch_size: int = 8):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.pending: List[Tuple[int, torch.Tensor, int]] = []  # (seq_id, input_ids, max_new_tokens)
        self.active: Dict[int, dict] = {}
        self._next_seq_id = 0

    def add_request(self, input_ids: torch.Tensor, max_new_tokens: int) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.pending.append((seq_id, input_ids, max_new_tokens))
        return seq_id

    def _admit_pending(self) -> None:
        while self.pending and len(self.active) < self.max_batch_size:
            seq_id, input_ids, max_new_tokens = self.pending.pop(0)
            next_token = self.engine.prefill(input_ids, seq_id)
            self.active[seq_id] = {
                "input_ids": input_ids,
                "cur_token": next_token,
                "generated": [next_token],
                "max_new_tokens": max_new_tokens,
                "num_generated": 1,
            }

    def step(self) -> Dict[int, torch.Tensor]:
        """Runs one iteration: admits pending requests if there's room,
        then runs one batched decode step across every active sequence.
        Returns {seq_id: full_token_sequence} for any sequence that
        finished this iteration (empty dict if none did)."""
        self._admit_pending()
        if not self.active:
            return {}

        seq_tokens = {sid: st["cur_token"] for sid, st in self.active.items()}
        next_tokens = self.engine.decode_step_batched(seq_tokens)

        finished_ids = []
        for sid, tok in next_tokens.items():
            st = self.active[sid]
            st["cur_token"] = tok
            st["generated"].append(tok)
            st["num_generated"] += 1
            if st["num_generated"] >= st["max_new_tokens"]:
                finished_ids.append(sid)

        results = {}
        for sid in finished_ids:
            st = self.active.pop(sid)
            results[sid] = torch.cat([st["input_ids"]] + st["generated"], dim=-1)
            self.engine.manager.free_sequence(sid)

        return results

    def run_to_completion(self) -> Dict[int, torch.Tensor]:
        """Runs step() repeatedly until every pending and active request
        has finished. Returns {seq_id: full_token_sequence} for all of
        them."""
        all_results: Dict[int, torch.Tensor] = {}
        while self.pending or self.active:
            all_results.update(self.step())
        return all_results
