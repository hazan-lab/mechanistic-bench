# headwise_stu (attn+STU) vs siblings

Run date: 2026-04-23. Sources: `/scratch/gpfs/EHAZAN/tharuntk/mech_runs/headwise_stu`, `/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard`.

Hypothesis under test: Mamba-2 hurt the headwise hybrid even though pure Mamba-2 beats pure Mamba-1 — maybe the Mamba kernel is a bad fit for a head-split and STU would pair better with attention.

## 1. Summary

- **1m shared-task means** (n=62 tasks with all four variants): headwise (m1) 0.495, headwise_mamba2 0.495, **headwise_stu 0.483**, pure stu 0.334.
- **headwise_stu vs headwise (m1) baseline:** Δmean tok_acc = -0.012; wins on 30/62 (48%) tasks.
- **headwise_stu vs headwise_mamba2:** Δmean tok_acc = -0.012; wins on 29/63 (46%) tasks.
- **headwise_stu vs pure STU:** Δmean tok_acc = +0.149; wins on 38/62 (61%) tasks. (If this is ≤ 0, the headwise split is not helping STU — it's just dragging pure STU down.)
- **1m extremes vs headwise-m1:** biggest gain `selective_copy` 0.027→0.197 (+0.170); biggest regression `last_tagged` 0.963→0.320 (-0.643).
- **10m shared-task means** (n=29): headwise_mamba2 0.188, **headwise_stu 0.183**, pure stu 0.163, attn 0.255.
- No `headwise` (Mamba-1) runs exist at 10m — that column is skipped in the 10m table.

## 2. 1m head-to-head

Final `tok_acc`. Sorted by (headwise_stu − headwise) descending — top rows are the tasks where swapping Mamba-1 for STU in the headwise split helped the most.
Bold `headwise_stu` cell: |Δ vs headwise-m1| ≥ 0.1.

| task | headwise (m1) | headwise_mamba2 | headwise_stu | Δ vs m1 | Δ vs m2 | pure stu | Δ vs pureSTU | attn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selective_copy | 0.027 | 0.056 | **0.197** | +0.170 | +0.141 | 0.057 | +0.140 | 0.670 |
| induction | 0.870 | 0.491 | 0.932 | +0.062 | +0.440 | 0.020 | +0.912 | 1.000 |
| nearest_key | 0.082 | 0.088 | 0.135 | +0.053 | +0.047 | 0.142 | -0.007 | 0.141 |
| counting | 0.913 | 0.230 | 0.965 | +0.052 | +0.734 | 0.151 | +0.813 | 0.991 |
| multi_induction | 0.879 | 0.721 | 0.917 | +0.038 | +0.196 | 0.022 | +0.895 | 1.000 |
| reverse_copy | 0.164 | 0.166 | 0.200 | +0.036 | +0.034 | 0.017 | +0.182 | 1.000 |
| union_lookup | 0.085 | 0.095 | 0.116 | +0.031 | +0.021 | 0.111 | +0.005 | 0.154 |
| state_retrieve | 0.957 | 0.991 | 0.979 | +0.022 | -0.012 | 0.622 | +0.357 | 0.988 |
| assignment_chain | 0.442 | 0.446 | 0.464 | +0.021 | +0.018 | 0.249 | +0.215 | 0.459 |
| set_intersection_count | 0.106 | 0.220 | 0.128 | +0.021 | -0.092 | 0.139 | -0.011 | 0.308 |
| associative | 0.052 | 0.056 | 0.072 | +0.021 | +0.017 | 0.091 | -0.019 | 0.101 |
| short_associative | 0.228 | 0.229 | 0.248 | +0.021 | +0.019 | 0.256 | -0.008 | 0.240 |
| substring_locate | 0.144 | 0.144 | 0.163 | +0.020 | +0.020 | 0.147 | +0.016 | 0.248 |
| quad_recall | 0.142 | 0.149 | 0.160 | +0.018 | +0.012 | 0.149 | +0.012 | 0.153 |
| triple_recall | 0.141 | 0.134 | 0.157 | +0.016 | +0.023 | 0.160 | -0.003 | 0.159 |
| first_vs_last | 0.249 | 0.415 | 0.263 | +0.014 | -0.152 | 0.363 | -0.101 | 0.421 |
| needle | 0.011 | 0.017 | 0.024 | +0.014 | +0.008 | 0.022 | +0.002 | 0.046 |
| noisy_copy | 0.026 | 0.035 | 0.035 | +0.009 | +0.000 | 0.054 | -0.019 | 0.118 |
| variable_lookup | 0.214 | 0.219 | 0.220 | +0.006 | +0.001 | 0.167 | +0.052 | 0.223 |
| nested_lookup | 0.340 | 0.339 | 0.345 | +0.005 | +0.006 | 0.336 | +0.009 | 0.333 |
| two_hop | 0.221 | 0.234 | 0.226 | +0.005 | -0.009 | 0.229 | -0.003 | 0.229 |
| conditional_recall | 0.157 | 0.141 | 0.161 | +0.004 | +0.021 | 0.068 | +0.093 | 0.159 |
| grid_two_coord | 0.024 | 0.022 | 0.028 | +0.004 | +0.006 | 0.040 | -0.011 | 0.118 |
| sort_top2 | 0.674 | 0.674 | 0.678 | +0.004 | +0.003 | 0.639 | +0.039 | 0.678 |
| three_hop | 0.259 | 0.278 | 0.263 | +0.004 | -0.016 | 0.077 | +0.186 | 0.275 |
| video_cell_mode | 0.222 | 0.071 | 0.226 | +0.004 | +0.154 | 0.200 | +0.025 | 0.919 |
| token_transition | 0.764 | 0.845 | 0.767 | +0.003 | -0.078 | 0.593 | +0.174 | 0.923 |
| k_hop | 0.280 | 0.280 | 0.282 | +0.002 | +0.002 | 0.181 | +0.102 | 0.275 |
| selective_parity | 0.120 | 0.121 | 0.121 | +0.001 | +0.000 | 0.140 | -0.019 | 0.999 |
| copy_count | 0.990 | 0.990 | 0.991 | +0.001 | +0.001 | 0.733 | +0.258 | 0.990 |
| compress | 1.000 | 1.000 | 1.000 | +0.000 | -0.000 | 0.149 | +0.850 | 0.999 |
| copy | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 0.017 | +0.983 | 1.000 |
| copy_offset | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 0.508 | +0.492 | 1.000 |
| cumulative_sum | 0.119 | 0.119 | 0.119 | +0.000 | +0.000 | 0.119 | +0.000 | 0.119 |
| delayed_echo | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 1.000 | +0.000 | 1.000 |
| hop_distance_bucket | 0.600 | 0.600 | 0.600 | +0.000 | +0.000 | 0.276 | +0.323 | 0.600 |
| mode | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 1.000 | +0.000 | 1.000 |
| mode_tagged | 0.890 | 0.886 | 0.890 | +0.000 | +0.004 | 0.889 | +0.001 | 0.889 |
| parity | 0.514 | 0.514 | 0.514 | +0.000 | +0.000 | 0.514 | +0.000 | 0.514 |
| patch_match | 0.496 | 0.496 | 0.496 | +0.000 | +0.000 | 0.496 | +0.000 | 0.496 |
| pattern_completion | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 1.000 | +0.000 | 1.000 |
| running_max | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 0.979 | +0.021 | 1.000 |
| threshold | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 1.000 | +0.000 | 1.000 |
| piecewise_denoise | 0.997 | 0.997 | 0.996 | -0.000 | -0.001 | 0.982 | +0.014 | 0.997 |
| grid_three_coord | 0.029 | 0.024 | 0.028 | -0.001 | +0.003 | 0.041 | -0.014 | 0.137 |
| deep_hop | 0.344 | 0.366 | 0.342 | -0.002 | -0.024 | 0.357 | -0.016 | 0.360 |
| col_parity | 0.462 | 0.462 | 0.459 | -0.003 | -0.003 | 0.462 | -0.003 | 0.462 |
| sort | 1.000 | 0.999 | 0.996 | -0.004 | -0.004 | 0.841 | +0.154 | 1.000 |
| nested_3_hop | 0.416 | 0.407 | 0.412 | -0.004 | +0.005 | 0.416 | -0.004 | 0.410 |
| video_frame_retrieval | 0.033 | 0.045 | 0.029 | -0.004 | -0.016 | 0.098 | -0.068 | 0.471 |
| longest_run | 1.000 | 1.000 | 0.995 | -0.005 | -0.005 | 0.703 | +0.292 | 1.000 |
| grid_retrieval | 0.031 | 0.020 | 0.025 | -0.006 | +0.006 | 0.043 | -0.018 | 0.151 |
| batch_two_hop | 0.327 | 0.366 | 0.320 | -0.007 | -0.046 | 0.317 | +0.003 | 0.329 |
| dual_query_hop | 0.287 | 0.842 | 0.271 | -0.015 | -0.570 | 0.189 | +0.083 | 0.276 |
| temporal_ordering | 0.507 | 0.483 | 0.483 | -0.023 | +0.000 | 0.535 | -0.052 | 0.735 |
| grid_multihop | 0.229 | 0.265 | 0.187 | -0.043 | -0.078 | 0.184 | +0.003 | 0.427 |
| multi_state_tracking | 0.973 | 0.990 | 0.930 | -0.043 | -0.061 | 0.441 | +0.488 | 0.876 |
| interleave | 0.434 | 0.332 | 0.391 | -0.044 | +0.058 | 0.108 | +0.283 | 1.000 |
| induction_gap | 0.890 | 0.889 | **0.761** | -0.129 | -0.128 | 0.015 | +0.746 | 1.000 |
| dual_hop_retrieve | 0.470 | 0.738 | **0.251** | -0.219 | -0.487 | 0.250 | +0.001 | 0.854 |
| state_tracking | 0.915 | 0.964 | **0.680** | -0.235 | -0.284 | 0.325 | +0.354 | 0.699 |
| last_tagged | 0.963 | 0.992 | **0.320** | -0.643 | -0.672 | 0.306 | +0.015 | 0.319 |
| batch_recall | — | 0.143 | 0.158 | — | +0.014 | — | — | — |

## 3. 1m aggregates

### Per-arch coverage and mean over all tasks present

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| headwise | 62 | 0.495 |
| headwise_mamba2 | 63 | 0.489 |
| headwise_stu | 63 | 0.478 |
| stu | 62 | 0.334 |
| attn | 62 | 0.588 |

### Shared-task (62 tasks, all four of headwise/headwise_mamba2/headwise_stu/stu present)

| arch | mean tok_acc on shared | Δ vs headwise (m1) |
|---|---:|---:|
| headwise | 0.495 | — |
| headwise_mamba2 | 0.495 | -0.000 |
| headwise_stu | 0.483 | -0.012 |
| stu | 0.334 | -0.161 |
| attn | 0.588 | +0.092 |

### Wins of `headwise_stu`

| opponent | wins / ties / n | win rate |
|---|---|---:|
| headwise (m1) | 30 / 13 / 62 | 48% |
| headwise_mamba2 | 29 / 13 / 63 | 46% |
| pure stu | 38 / 7 / 62 | 61% |

## 4. 10m head-to-head

Final `tok_acc`. Sorted by (headwise_stu − headwise_mamba2) descending. No `headwise` (Mamba-1) at 10m.

| task | headwise_mamba2 | headwise_stu | Δ vs m2 | pure stu | Δ vs pureSTU | attn | lstm |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_vs_last | 0.078 | 0.134 | +0.056 | 0.082 | +0.052 | 0.461 | 0.080 |
| assignment_chain | 0.226 | 0.280 | +0.055 | 0.174 | +0.106 | 0.324 | 0.297 |
| set_intersection_count | 0.070 | 0.124 | +0.054 | 0.082 | +0.042 | 0.244 | 0.166 |
| two_hop | 0.060 | 0.113 | +0.054 | 0.123 | -0.010 | 0.119 | 0.111 |
| three_hop | 0.126 | 0.174 | +0.048 | 0.039 | +0.135 | 0.131 | 0.119 |
| deep_hop | 0.153 | 0.196 | +0.043 | 0.057 | +0.140 | 0.260 | 0.268 |
| dual_query_hop | 0.083 | 0.112 | +0.029 | 0.101 | +0.011 | 0.140 | 0.143 |
| video_cell_mode | 0.044 | 0.072 | +0.028 | 0.139 | -0.066 | 0.975 | 0.080 |
| dual_hop_retrieve | 0.077 | 0.102 | +0.024 | 0.126 | -0.024 | 0.113 | 0.091 |
| triple_recall | 0.082 | 0.098 | +0.016 | 0.092 | +0.006 | 0.095 | 0.094 |
| nested_lookup | 0.181 | 0.196 | +0.016 | 0.164 | +0.032 | 0.184 | 0.184 |
| k_hop | 0.166 | 0.181 | +0.015 | 0.117 | +0.063 | 0.186 | 0.186 |
| mode_tagged | 0.818 | 0.829 | +0.011 | 0.824 | +0.005 | 0.914 | 0.830 |
| quad_recall | 0.087 | 0.097 | +0.010 | 0.096 | +0.001 | 0.098 | 0.100 |
| grid_two_coord | 0.024 | 0.033 | +0.009 | 0.038 | -0.005 | 0.042 | 0.033 |
| sort_top2 | 0.562 | 0.568 | +0.007 | 0.479 | +0.089 | 0.559 | 0.558 |
| conditional_recall | 0.052 | 0.058 | +0.006 | 0.047 | +0.011 | 0.113 | 0.098 |
| video_frame_retrieval | 0.034 | 0.039 | +0.005 | 0.045 | -0.006 | 0.025 | 0.066 |
| variable_lookup | 0.109 | 0.110 | +0.001 | 0.091 | +0.019 | 0.114 | 0.111 |
| grid_three_coord | 0.028 | 0.029 | +0.001 | 0.043 | -0.014 | 0.043 | 0.031 |
| substring_locate | 0.074 | 0.074 | +0.000 | 0.102 | -0.027 | 0.062 | 0.098 |
| temporal_ordering | 0.491 | 0.491 | +0.000 | 0.504 | -0.013 | 0.574 | 0.504 |
| batch_recall | 0.096 | 0.093 | -0.003 | 0.095 | -0.002 | 0.096 | 0.092 |
| batch_two_hop | 0.180 | 0.176 | -0.004 | 0.179 | -0.003 | 0.183 | 0.190 |
| nested_3_hop | 0.237 | 0.225 | -0.013 | 0.264 | -0.039 | 0.256 | 0.252 |
| union_lookup | 0.040 | 0.025 | -0.015 | 0.062 | -0.037 | 0.088 | 0.104 |
| grid_multihop | 0.312 | 0.287 | -0.024 | 0.168 | +0.119 | 0.430 | 0.217 |
| hop_distance_bucket | 0.395 | **0.231** | -0.163 | 0.236 | -0.005 | 0.408 | 0.408 |
| last_tagged | 0.562 | **0.146** | -0.416 | 0.160 | -0.015 | 0.168 | 0.180 |

## 5. 10m aggregates

### Per-arch coverage and mean over all tasks present

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| headwise_mamba2 | 29 | 0.188 |
| headwise_stu | 29 | 0.183 |
| stu | 29 | 0.163 |
| attn | 29 | 0.255 |
| lstm | 29 | 0.196 |

### Shared-task (29 tasks, headwise_mamba2/headwise_stu/stu/attn all present)

| arch | mean tok_acc on shared |
|---|---:|
| headwise_mamba2 | 0.188 |
| headwise_stu | 0.183 |
| stu | 0.163 |
| attn | 0.255 |
| lstm | 0.196 |

### Wins of `headwise_stu` at 10m

| opponent | wins / ties / n | win rate |
|---|---|---:|
| headwise_mamba2 | 20 / 2 / 29 | 69% |
| pure stu | 15 / 0 / 29 | 52% |
| attn | 6 / 0 / 29 | 21% |

## 6. Takeaway

- **STU is roughly a wash vs Mamba-1 inside the headwise split** (Δmean -0.012). Kernel choice is not the dominant factor; the head-split shape is.
- Pairing STU with attention in a head-split **improves on pure STU** (Δmean +0.149) — the hybrid is doing real work.
- STU ≈ Mamba-2 inside the head-split (Δmean -0.012).

**Bottom line:** mixed — see the per-comparison bullets above. The headwise split is not cleanly rescued by STU, but it is also not uniformly harmful.

## 7. Notes

### Anomalies / collapsed runs

No NaN or collapsed (`tok_acc ≤ 0.01`) runs detected across the four stores.

CSVs: `/home/tt6444/mechanistic-bench/figures/headwise_stu_1m.csv`, `/home/tt6444/mechanistic-bench/figures/headwise_stu_10m.csv`.
