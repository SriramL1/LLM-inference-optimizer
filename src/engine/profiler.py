"""
Profiling instrumentation for the inference engine.

Measures the two numbers that matter for LLM inference performance:
  - TTFT (time to first token): prefill latency, compute-bound
  - Decode tokens/sec: steady-state per-token throughput once the KV-cache
    is warm, memory-bandwidth-bound

They're tracked separately on purpose -- prefill and decode stress
different parts of the GPU, which is why the whole optimization roadmap
(fused attention for prefill; paged KV-cache and continuous batching for
decode) splits along this line.

Uses torch.cuda.Event for GPU-side timing (wall-clock time.time() would
include CPU-side dispatch overhead and give misleading kernel timings).
"""
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import torch


@dataclass
class RunResult:
    model_name: str
    prompt_tokens: int
    max_new_tokens: int
    batch_size: int
    ttft_ms: float                  # time to first token (prefill)
    decode_tokens_per_sec: float    # steady-state decode throughput
    total_tokens_per_sec: float     # end-to-end (prefill + decode)
    peak_memory_mb: float
    per_token_latencies_ms: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregatedResult:
    """Mean/std across N trials of the same config.

    Single-shot timings are noisy -- background GPU load (desktop
    compositor, browser hardware acceleration, other apps) shows up
    disproportionately at low batch sizes / low GPU utilization. Reporting
    a spread, not just one number, is what makes a later stage's speedup
    claim ("Stage 2 cut TTFT by 20%") trustworthy rather than noise.
    """
    model_name: str
    prompt_tokens: int
    max_new_tokens: int
    batch_size: int
    n_trials: int
    ttft_ms_mean: float
    ttft_ms_std: float
    decode_tokens_per_sec_mean: float
    decode_tokens_per_sec_std: float
    total_tokens_per_sec_mean: float
    total_tokens_per_sec_std: float
    peak_memory_mb_mean: float
    trials: List[RunResult] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


class InferenceProfiler:
    """Wraps CUDA event timing + memory tracking around a generation run."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def _event(self):
        return torch.cuda.Event(enable_timing=True)

    def profile_generation(
        self,
        engine,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        model_name: str,
    ) -> RunResult:
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        batch_size, prompt_len = input_ids.shape

        # --- Prefill ---
        if use_cuda:
            start_prefill, end_prefill = self._event(), self._event()
            start_prefill.record()
            past_key_values, next_token = engine.prefill(input_ids)
            end_prefill.record()
            torch.cuda.synchronize()
            ttft_ms = start_prefill.elapsed_time(end_prefill)
        else:
            t0 = time.perf_counter()
            past_key_values, next_token = engine.prefill(input_ids)
            ttft_ms = (time.perf_counter() - t0) * 1000

        # --- Decode (timed per-step so we can see tail latency, not just
        # the average) ---
        generated = [next_token]
        step_times_ms = []
        cur_token = next_token

        if use_cuda:
            decode_start, decode_end = self._event(), self._event()
            decode_start.record()
            for _ in range(max_new_tokens - 1):
                e_start, e_end = self._event(), self._event()
                e_start.record()
                past_key_values, cur_token = engine.decode_step(cur_token, past_key_values)
                e_end.record()
                step_times_ms.append((e_start, e_end))
                generated.append(cur_token)
            decode_end.record()
            torch.cuda.synchronize()
            per_token_latencies_ms = [s.elapsed_time(e) for s, e in step_times_ms]
            decode_time_ms = decode_start.elapsed_time(decode_end)
        else:
            decode_t0 = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                s = time.perf_counter()
                past_key_values, cur_token = engine.decode_step(cur_token, past_key_values)
                step_times_ms.append((time.perf_counter() - s) * 1000)
                generated.append(cur_token)
            decode_time_ms = (time.perf_counter() - decode_t0) * 1000
            per_token_latencies_ms = step_times_ms

        n_decode_tokens = len(generated) - 1  # first token already counted in prefill

        decode_tokens_per_sec = (
            (n_decode_tokens * batch_size) / (decode_time_ms / 1000)
            if decode_time_ms > 0 and n_decode_tokens > 0
            else 0.0
        )
        total_time_s = (ttft_ms + decode_time_ms) / 1000
        total_tokens_per_sec = (len(generated) * batch_size) / total_time_s

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if use_cuda else 0.0
        )

        return RunResult(
            model_name=model_name,
            prompt_tokens=prompt_len,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=decode_tokens_per_sec,
            total_tokens_per_sec=total_tokens_per_sec,
            peak_memory_mb=peak_memory_mb,
            per_token_latencies_ms=per_token_latencies_ms,
        )

    def profile_generation_repeated(
        self,
        engine,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        model_name: str,
        n_trials: int = 5,
    ) -> AggregatedResult:
        """Run the same config n_trials times and report mean/std.

        Each trial is a fully independent measurement (fresh peak-memory
        reset, fresh CUDA-event timing) so trial-to-trial variance reflects
        real system noise (thermal/clock state, background GPU load from
        other apps), not measurement artifacts.
        """
        trials = [
            self.profile_generation(engine, input_ids, max_new_tokens, model_name)
            for _ in range(n_trials)
        ]

        def _mean(attr):
            return statistics.mean(getattr(t, attr) for t in trials)

        def _std(attr):
            return statistics.stdev(getattr(t, attr) for t in trials) if n_trials > 1 else 0.0

        return AggregatedResult(
            model_name=model_name,
            prompt_tokens=trials[0].prompt_tokens,
            max_new_tokens=max_new_tokens,
            batch_size=trials[0].batch_size,
            n_trials=n_trials,
            ttft_ms_mean=_mean("ttft_ms"),
            ttft_ms_std=_std("ttft_ms"),
            decode_tokens_per_sec_mean=_mean("decode_tokens_per_sec"),
            decode_tokens_per_sec_std=_std("decode_tokens_per_sec"),
            total_tokens_per_sec_mean=_mean("total_tokens_per_sec"),
            total_tokens_per_sec_std=_std("total_tokens_per_sec"),
            peak_memory_mb_mean=_mean("peak_memory_mb"),
            trials=trials,
        )


def save_results(results: List[AggregatedResult], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def print_summary_table(results: List[AggregatedResult]):
    header = (
        f"{'model':<28}{'prompt':>8}{'batch':>7}{'gen':>6}{'n':>4}"
        f"{'TTFT(ms)':>14}{'decode tok/s':>18}{'total tok/s':>17}{'peak MB':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ttft_str = f"{r.ttft_ms_mean:.1f}±{r.ttft_ms_std:.1f}"
        decode_str = f"{r.decode_tokens_per_sec_mean:.1f}±{r.decode_tokens_per_sec_std:.1f}"
        total_str = f"{r.total_tokens_per_sec_mean:.1f}±{r.total_tokens_per_sec_std:.1f}"
        print(
            f"{r.model_name:<28}{r.prompt_tokens:>8}{r.batch_size:>7}{r.max_new_tokens:>6}{r.n_trials:>4}"
            f"{ttft_str:>14}{decode_str:>18}{total_str:>17}{r.peak_memory_mb_mean:>10.1f}"
        )
