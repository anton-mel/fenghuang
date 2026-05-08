"""
FengHuang TAB Simulator v2 — corrected methodology
====================================================
Uses MEASURED timing from both traces:
  - Baseline8 (TP=8, BF16): trace_baseline8.sqlite + server log throughput
  - FH4 (TP=4, FP8): trace_fh4_qa.sqlite + server log throughput

My hardware: H100 80GB SXM5 (fp16 peak 989 TFLOPS, bw 3.35 TB/s)
Paper hardware: H200 141GB (fp8 peak 1979 TFLOPS, bw 4.8 TB/s)
"""

import sqlite3
import json
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Measured server-log throughput (tokens/sec total across all 8 requests)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE8_TOKS_PER_SEC  = 780.0   # steady-state from server log
FH4_TOKS_PER_SEC        = 675.0   # steady-state from server log
N_REQUESTS              = 8
OUTPUT_TOKENS           = 1024    # max_new_tokens in paper workload

# Derived per-request TPOT (microseconds)
BASELINE8_TPOT_US = 1e6 / (BASELINE8_TOKS_PER_SEC / N_REQUESTS)
FH4_TPOT_US       = 1e6 / (FH4_TOKS_PER_SEC / N_REQUESTS)

PAPER_LOCAL_HBM_GB      = 20.0    # GB of local HBM in paper FH4 config
FH4_TP                  = 4
MODEL_PARAMS_B          = 235.0   # billion params
FP8_BYTES_PER_PARAM     = 1.0
MODEL_SIZE_GB           = MODEL_PARAMS_B * FP8_BYTES_PER_PARAM  # 235 GB
PER_GPU_SHARD_GB        = MODEL_SIZE_GB / FH4_TP                # 58.75 GB
REMOTE_BYTES_PER_STEP   = max(0, PER_GPU_SHARD_GB - PAPER_LOCAL_HBM_GB) * 1e9  # bytes

# Local HBM bandwidth (H200 in paper, H100 on Yale)
LOCAL_HBM_BW_GBs        = 3350.0  # H100 HBM3 measured

# Bandwidth efficiency (empirical, §4.1.3)
def bw_efficiency(size_bytes: float) -> float:
    curve = [
        (0,           0.10),
        (1 << 10,     0.30),
        (1 << 16,     0.60),
        (1 << 20,     0.85),
        (1 << 23,     0.95),
        (1 << 26,     1.00),
    ]
    for i, (thr, eff) in enumerate(curve):
        if size_bytes <= thr:
            if i == 0: return eff
            pt, pe = curve[i-1]
            t = (size_bytes - pt) / (thr - pt + 1e-9)
            return pe + t * (eff - pe)
    return curve[-1][1]


def get_ttft_from_trace(sqlite_path: str, inference_window_sec: float) -> float:
    """
    Extract TTFT from an nsys trace.
    """
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute('SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL')
    t_min = cur.fetchone()[0]
    
    window_start = t_min + int(inference_window_sec * 1e9)
    
    cur.execute('''SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ?''',
                (window_start,))
    inf_start = cur.fetchone()[0]
    
    # 100ms bucket kernel counts
    cur.execute('''
        SELECT (start - ?) / 100000000 as b, COUNT(*) as cnt
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ?
        GROUP BY b
        ORDER BY b
    ''', (inf_start, window_start))
    buckets = cur.fetchall()
    con.close()
    
    if not buckets:
        return 1000.0  # fallback 1 second
    
    # The first few buckets are prefill (high per-bucket count but brief),
    # then decode is sustained. Find first sustained bucket.
    counts = [c for _, c in buckets]
    median = sorted(counts)[len(counts)//2]
    
    # TTFT = time until first sustained decode activity
    # Sustained = 3 consecutive buckets all > 30% of median
    ttft_ms = 0.0
    for i, (b, c) in enumerate(buckets):
        if i >= 2:
            if (buckets[i-2][1] > median*0.3 and 
                buckets[i-1][1] > median*0.3 and 
                c > median*0.3):
                ttft_ms = buckets[i-2][0] * 100.0
                break
    
    return max(ttft_ms, 100.0)  # at least 100ms


def simulate_fh4_tab(fh4_tpot_us: float, 
                     remote_bw_GBs: float,
                     remote_bytes: float) -> dict:
    """
    Simulate FH4 + TAB improvement over measured FH4.
    """
    # Time to stream weights from local HBM (without TAB)
    local_stall_ns = remote_bytes / LOCAL_HBM_BW_GBs * 1e9 / 1e9  # ns
    local_stall_us = local_stall_ns / 1e3
    
    # FH4 compute-only TPOT (subtracting estimated local memory stall)
    compute_tpot_us = max(fh4_tpot_us - local_stall_us * 0.3, fh4_tpot_us * 0.5)
    
    # With TAB at remote_bw_GBs:
    eff = bw_efficiency(remote_bytes)
    tab_fetch_ns = remote_bytes / (remote_bw_GBs * eff)  # seconds × 1e9 = ns... 
    # Actually: tab_fetch_ns = remote_bytes [bytes] / (remote_bw_GBs [GB/s] × 1e9 [B/GB])
    tab_fetch_us = (remote_bytes / (remote_bw_GBs * 1e9)) * 1e6  # microseconds
    tab_fetch_us /= eff
    
    # Stall = max(0, tab_fetch_us - compute_tpot_us)
    stall_us = max(0.0, tab_fetch_us - compute_tpot_us)
    tab_tpot_us = compute_tpot_us + stall_us
    
    return {
        'compute_tpot_us':   compute_tpot_us,
        'tab_fetch_us':      tab_fetch_us,
        'stall_us':          stall_us,
        'tab_tpot_us':       tab_tpot_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-trace', required=True)
    parser.add_argument('--fh4-trace',      required=True)
    parser.add_argument('--output',          default='results_corrected.json')
    parser.add_argument('--bw-sweep',        nargs='+', type=float, default=[4.0, 4.8, 6.4])
    args = parser.parse_args()

    print('=== FengHuang TAB Simulator v2 ===')
    print(f'Measured Baseline8 TPOT: {BASELINE8_TPOT_US:.0f} µs/token/req')
    print(f'Measured FH4 TPOT:       {FH4_TPOT_US:.0f} µs/token/req')
    print(f'Remote weight bytes/step (paper scenario): {REMOTE_BYTES_PER_STEP/1e9:.2f} GB')

    # TTFT from traces
    print('\nExtracting TTFT from traces...')
    baseline_ttft_ms = get_ttft_from_trace(args.baseline_trace, 91.5)
    fh4_ttft_ms      = get_ttft_from_trace(args.fh4_trace,      165.5)
    print(f'  Baseline8 TTFT: {baseline_ttft_ms:.0f} ms')
    print(f'  FH4 TTFT:       {fh4_ttft_ms:.0f} ms')

    results = []

    # Baseline8
    b8_e2e = baseline_ttft_ms + BASELINE8_TPOT_US * OUTPUT_TOKENS / 1e3
    baseline = {
        'system':    'Baseline8 (NVLink)',
        'ttft_ms':   baseline_ttft_ms,
        'tpot_us':   BASELINE8_TPOT_US,
        'e2e_ms':    b8_e2e,
    }
    results.append(baseline)
    print(f'\nBaseline8: TTFT={baseline_ttft_ms:.0f}ms, TPOT={BASELINE8_TPOT_US:.0f}µs, E2E={b8_e2e:.0f}ms')

    # FH4 with TAB sweep
    for bw_TBs in args.bw_sweep:
        bw_GBs = bw_TBs * 1000
        sim = simulate_fh4_tab(FH4_TPOT_US, bw_GBs, REMOTE_BYTES_PER_STEP)
        
        tpot_us  = sim['tab_tpot_us']
        stall_ms = sim['stall_us'] * OUTPUT_TOKENS / 1e3
        e2e_ms   = fh4_ttft_ms + tpot_us * OUTPUT_TOKENS / 1e3
        
        r = {
            'system':         f'FH4 @ {bw_TBs} TB/s',
            'remote_bw_GBs':  bw_GBs,
            'ttft_ms':        fh4_ttft_ms,
            'tpot_us':        tpot_us,
            'e2e_ms':         e2e_ms,
            'stall_ms':       stall_ms,
        }
        results.append(r)
        print(f'FH4 @ {bw_TBs} TB/s: TPOT={tpot_us:.0f}µs '
              f'(stall={sim["stall_us"]:.0f}µs, fetch={sim["tab_fetch_us"]:.0f}µs), '
              f'E2E={e2e_ms:.0f}ms')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults → {args.output}')


if __name__ == '__main__':
    main()
