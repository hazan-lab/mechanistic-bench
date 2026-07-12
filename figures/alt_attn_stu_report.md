# alt_attn_stu (layer-wise attn+STU hybrid) vs siblings

Run date: 2026-04-23. Data dirs: `/scratch/gpfs/EHAZAN/tharuntk/mech_runs/alt_attn_stu`, `/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard`.

## 1. Summary

- **1m intersection (62 tasks), mean tok_acc:** attn 0.588 | alt_attn_mamba 0.682 | alt_attn_mamba2 0.618 | **alt_attn_stu 0.652** | pure stu 0.334.
- **alt_attn_stu vs alt_attn_mamba (prior-best hybrid):** Δmean -0.029; wins 19/62 (ties 21) tasks.
- **alt_attn_stu vs pure attn:** Δmean +0.064; wins 32/62 (ties 16) tasks.
- **alt_attn_stu vs pure stu:** Δmean +0.318; wins 51/62 (ties 8) tasks.
- **alt_attn_stu vs alt_attn_mamba2:** Δmean +0.034; wins 24/63 (ties 16) tasks.
- **10m intersection (29 tasks):** attn 0.255 | alt_attn_mamba2 0.265 | **alt_attn_stu 0.293** | pure stu 0.163 | lstm 0.196. (No alt_attn_mamba at 10m.)

## 2. 1m head-to-head (per task)

Columns: `attn | alt_attn_mamba | alt_attn_mamba2 | alt_attn_stu | pure stu`. Δ columns in the right block: `alt_attn_stu − alt_attn_mamba` (sort key) and `alt_attn_stu − attn`. Sorted by `alt_attn_stu − alt_attn_mamba` desc. Bold entries have |Δ vs alt_attn_mamba| ≥ 0.1.

| task | attn | alt_attn_mamba | alt_attn_mamba2 | alt_attn_stu | pure stu | Δ stu−m1 | Δ stu−attn |
|---|---:|---:|---:|---:|---:|---:|---:|
| set_intersection_count | 0.308 | 0.461 | 0.447 | **0.998** | 0.139 | +0.537 | +0.690 |
| temporal_ordering | 0.735 | 0.483 | 0.609 | **0.793** | 0.535 | +0.310 | +0.058 |
| substring_locate | 0.248 | 0.191 | 0.366 | **0.327** | 0.147 | +0.136 | +0.079 |
| state_tracking | 0.699 | 0.883 | 0.934 | 0.973 | 0.325 | +0.090 | +0.273 |
| noisy_copy | 0.118 | 0.923 | 0.839 | 0.993 | 0.054 | +0.070 | +0.875 |
| dual_hop_retrieve | 0.854 | 0.933 | 0.985 | 1.000 | 0.250 | +0.067 | +0.146 |
| grid_multihop | 0.427 | 0.363 | 0.451 | 0.398 | 0.184 | +0.035 | -0.028 |
| nested_3_hop | 0.410 | 0.404 | 0.404 | 0.420 | 0.416 | +0.016 | +0.010 |
| variable_lookup | 0.223 | 0.213 | 0.221 | 0.227 | 0.167 | +0.015 | +0.004 |
| union_lookup | 0.154 | 0.152 | 0.153 | 0.159 | 0.111 | +0.007 | +0.005 |
| counting | 0.991 | 0.995 | 0.425 | 0.999 | 0.151 | +0.004 | +0.008 |
| nearest_key | 0.141 | 0.141 | 0.148 | 0.145 | 0.142 | +0.004 | +0.004 |
| copy_count | 0.990 | 0.990 | 0.992 | 0.993 | 0.733 | +0.004 | +0.004 |
| needle | 0.046 | 0.032 | 0.037 | 0.035 | 0.022 | +0.003 | -0.011 |
| triple_recall | 0.159 | 0.154 | 0.153 | 0.156 | 0.160 | +0.002 | -0.003 |
| selective_copy | 0.670 | 0.675 | 0.677 | 0.676 | 0.057 | +0.001 | +0.006 |
| reverse_copy | 1.000 | 0.999 | 1.000 | 1.000 | 0.017 | +0.001 | +0.000 |
| sort_top2 | 0.678 | 0.677 | 0.678 | 0.677 | 0.639 | +0.000 | -0.001 |
| piecewise_denoise | 0.997 | 0.997 | 0.997 | 0.997 | 0.982 | +0.000 | -0.000 |
| col_parity | 0.462 | 0.462 | 0.462 | 0.462 | 0.462 | +0.000 | +0.000 |
| compress | 0.999 | 1.000 | 1.000 | 1.000 | 0.149 | +0.000 | +0.001 |
| copy | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 | +0.000 | +0.000 |
| copy_offset | 1.000 | 1.000 | 1.000 | 1.000 | 0.508 | +0.000 | +0.000 |
| cumulative_sum | 0.119 | 0.119 | 0.119 | 0.119 | 0.119 | +0.000 | +0.000 |
| delayed_echo | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| grid_retrieval | 0.151 | 1.000 | 0.104 | 1.000 | 0.043 | +0.000 | +0.849 |
| induction | 1.000 | 1.000 | 1.000 | 1.000 | 0.020 | +0.000 | +0.000 |
| induction_gap | 1.000 | 1.000 | 1.000 | 1.000 | 0.015 | +0.000 | +0.000 |
| interleave | 1.000 | 1.000 | 1.000 | 1.000 | 0.108 | +0.000 | +0.000 |
| longest_run | 1.000 | 1.000 | 1.000 | 1.000 | 0.703 | +0.000 | +0.000 |
| mode | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| mode_tagged | 0.889 | 1.000 | 1.000 | 1.000 | 0.889 | +0.000 | +0.111 |
| multi_induction | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | +0.000 | +0.000 |
| parity | 0.514 | 0.514 | 0.514 | 0.514 | 0.514 | +0.000 | +0.000 |
| patch_match | 0.496 | 0.496 | 0.496 | 0.496 | 0.496 | +0.000 | +0.000 |
| pattern_completion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| running_max | 1.000 | 1.000 | 1.000 | 1.000 | 0.979 | +0.000 | +0.000 |
| state_retrieve | 0.988 | 0.998 | 0.999 | 0.998 | 0.622 | +0.000 | +0.010 |
| threshold | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| video_frame_retrieval | 0.471 | 1.000 | 0.237 | 1.000 | 0.098 | +0.000 | +0.529 |
| sort | 1.000 | 1.000 | 1.000 | 1.000 | 0.841 | -0.000 | -0.000 |
| selective_parity | 0.999 | 1.000 | 1.000 | 1.000 | 0.140 | -0.000 | +0.001 |
| dual_query_hop | 0.276 | 0.280 | 0.999 | 0.280 | 0.189 | -0.000 | +0.004 |
| conditional_recall | 0.159 | 0.166 | 0.168 | 0.165 | 0.068 | -0.001 | +0.006 |
| short_associative | 0.240 | 0.251 | 0.244 | 0.250 | 0.256 | -0.001 | +0.010 |
| k_hop | 0.275 | 0.297 | 0.285 | 0.295 | 0.181 | -0.002 | +0.020 |
| quad_recall | 0.153 | 0.152 | 0.147 | 0.150 | 0.149 | -0.002 | -0.003 |
| nested_lookup | 0.333 | 0.343 | 0.342 | 0.340 | 0.336 | -0.003 | +0.007 |
| associative | 0.101 | 0.103 | 0.101 | 0.099 | 0.091 | -0.004 | -0.002 |
| token_transition | 0.923 | 0.991 | 0.955 | 0.977 | 0.593 | -0.015 | +0.054 |
| two_hop | 0.229 | 0.241 | 0.230 | 0.225 | 0.229 | -0.017 | -0.005 |
| hop_distance_bucket | 0.600 | 0.621 | 0.624 | 0.600 | 0.276 | -0.021 | +0.000 |
| three_hop | 0.275 | 0.296 | 0.269 | 0.273 | 0.077 | -0.022 | -0.002 |
| assignment_chain | 0.459 | 0.460 | 0.454 | 0.432 | 0.249 | -0.028 | -0.027 |
| batch_two_hop | 0.329 | 0.434 | 0.431 | 0.405 | 0.317 | -0.029 | +0.076 |
| multi_state_tracking | 0.876 | 0.968 | 0.970 | 0.931 | 0.441 | -0.037 | +0.055 |
| video_cell_mode | 0.919 | 0.996 | 0.380 | 0.952 | 0.200 | -0.044 | +0.033 |
| deep_hop | 0.360 | 0.442 | 0.443 | 0.367 | 0.357 | -0.075 | +0.007 |
| first_vs_last | 0.421 | 0.967 | 0.576 | **0.409** | 0.363 | -0.558 | -0.012 |
| grid_three_coord | 0.137 | 1.000 | 0.158 | **0.354** | 0.041 | -0.646 | +0.217 |
| last_tagged | 0.319 | 1.000 | 1.000 | **0.317** | 0.306 | -0.683 | -0.002 |
| grid_two_coord | 0.118 | 1.000 | 0.094 | **0.059** | 0.040 | -0.941 | -0.059 |
| batch_recall | — | — | 0.157 | 0.152 | — | — | — |

## 3. 1m aggregates

**Intersection basis** (62 tasks with all five columns):

| arch | n | mean tok_acc |
|---|---:|---:|
| attn | 62 | 0.588 |
| alt_attn_mamba | 62 | 0.682 |
| alt_attn_mamba2 | 62 | 0.618 |
| alt_attn_stu | 62 | 0.652 |
| stu | 62 | 0.334 |

**All-task basis** (per-arch coverage, mean over available tasks):

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| attn | 62 | 0.588 |
| alt_attn_mamba | 62 | 0.682 |
| alt_attn_mamba2 | 63 | 0.611 |
| alt_attn_stu | 63 | 0.644 |
| stu | 62 | 0.334 |

**Win counts (alt_attn_stu beats X on pairwise-present tasks):**

| comparison | wins / n |
|---|---:|
| alt_attn_stu vs alt_attn_mamba | 19/62 (ties 21) |
| alt_attn_stu vs alt_attn_mamba2 | 24/63 (ties 16) |
| alt_attn_stu vs pure stu | 51/62 (ties 8) |
| alt_attn_stu vs pure attn | 32/62 (ties 16) |

**Extremes at 1m:**

- Biggest gain vs alt_attn_mamba: `set_intersection_count` 0.461 → 0.998 (+0.537).
- Biggest regression vs alt_attn_mamba: `grid_two_coord` 1.000 → 0.059 (-0.941).
- Biggest gain vs pure attn: `noisy_copy` 0.118 → 0.993 (+0.875).
- Biggest regression vs pure attn: `grid_two_coord` 0.118 → 0.059 (-0.059).
- Biggest gain vs pure stu: `induction_gap` 0.015 → 1.000 (+0.985).
- Biggest regression vs pure stu: `short_associative` 0.256 → 0.250 (-0.006).

## 4. 10m head-to-head (per task)

Columns: `attn | alt_attn_mamba2 | alt_attn_stu | pure stu | lstm`. Sorted by `alt_attn_stu − alt_attn_mamba2` desc. Bold entries have |Δ vs alt_attn_mamba2| ≥ 0.1. No `alt_attn_mamba` (Mamba-1) runs exist at 10m.

| task | attn | alt_attn_mamba2 | alt_attn_stu | pure stu | lstm | Δ stu−m2 | Δ stu−attn |
|---|---:|---:|---:|---:|---:|---:|---:|
| set_intersection_count | 0.244 | 0.177 | **0.867** | 0.082 | 0.166 | +0.690 | +0.623 |
| video_frame_retrieval | 0.025 | 0.077 | **0.467** | 0.045 | 0.066 | +0.390 | +0.441 |
| temporal_ordering | 0.574 | 0.519 | **0.718** | 0.504 | 0.504 | +0.199 | +0.144 |
| first_vs_last | 0.461 | 0.272 | **0.416** | 0.082 | 0.080 | +0.144 | -0.045 |
| video_cell_mode | 0.975 | 0.791 | **0.897** | 0.139 | 0.080 | +0.106 | -0.077 |
| grid_multihop | 0.430 | 0.307 | 0.397 | 0.168 | 0.217 | +0.091 | -0.032 |
| grid_three_coord | 0.043 | 0.071 | 0.143 | 0.043 | 0.031 | +0.072 | +0.100 |
| grid_two_coord | 0.042 | 0.061 | 0.103 | 0.038 | 0.033 | +0.042 | +0.061 |
| assignment_chain | 0.324 | 0.280 | 0.300 | 0.174 | 0.297 | +0.020 | -0.024 |
| batch_recall | 0.096 | 0.094 | 0.098 | 0.095 | 0.092 | +0.004 | +0.002 |
| deep_hop | 0.260 | 0.211 | 0.215 | 0.057 | 0.268 | +0.004 | -0.045 |
| sort_top2 | 0.559 | 0.564 | 0.568 | 0.479 | 0.558 | +0.004 | +0.009 |
| k_hop | 0.186 | 0.182 | 0.185 | 0.117 | 0.186 | +0.003 | -0.001 |
| nested_3_hop | 0.256 | 0.233 | 0.236 | 0.264 | 0.252 | +0.003 | -0.020 |
| quad_recall | 0.098 | 0.095 | 0.098 | 0.096 | 0.100 | +0.002 | -0.000 |
| triple_recall | 0.095 | 0.097 | 0.098 | 0.092 | 0.094 | +0.002 | +0.003 |
| mode_tagged | 0.914 | 0.826 | 0.827 | 0.824 | 0.830 | +0.001 | -0.087 |
| two_hop | 0.119 | 0.132 | 0.133 | 0.123 | 0.111 | +0.001 | +0.014 |
| dual_query_hop | 0.140 | 0.124 | 0.125 | 0.101 | 0.143 | +0.000 | -0.015 |
| hop_distance_bucket | 0.408 | 0.395 | 0.395 | 0.236 | 0.408 | +0.000 | -0.014 |
| union_lookup | 0.088 | 0.083 | 0.081 | 0.062 | 0.104 | -0.002 | -0.007 |
| variable_lookup | 0.114 | 0.121 | 0.118 | 0.091 | 0.111 | -0.003 | +0.004 |
| batch_two_hop | 0.183 | 0.182 | 0.177 | 0.179 | 0.190 | -0.004 | -0.006 |
| three_hop | 0.131 | 0.160 | 0.155 | 0.039 | 0.119 | -0.005 | +0.024 |
| dual_hop_retrieve | 0.113 | 0.126 | 0.120 | 0.126 | 0.091 | -0.006 | +0.006 |
| conditional_recall | 0.113 | 0.101 | 0.094 | 0.047 | 0.098 | -0.007 | -0.020 |
| nested_lookup | 0.184 | 0.206 | 0.188 | 0.164 | 0.184 | -0.018 | +0.005 |
| substring_locate | 0.062 | 0.211 | 0.113 | 0.102 | 0.098 | -0.098 | +0.051 |
| last_tagged | 0.168 | 0.999 | **0.167** | 0.160 | 0.180 | -0.832 | -0.001 |

## 5. 10m aggregates

**Intersection basis** (29 tasks with all five columns):

| arch | n | mean tok_acc |
|---|---:|---:|
| attn | 29 | 0.255 |
| alt_attn_mamba2 | 29 | 0.265 |
| alt_attn_stu | 29 | 0.293 |
| stu | 29 | 0.163 |
| lstm | 29 | 0.196 |

**All-task basis:**

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| attn | 29 | 0.255 |
| alt_attn_mamba2 | 29 | 0.265 |
| alt_attn_stu | 29 | 0.293 |
| stu | 29 | 0.163 |
| lstm | 29 | 0.196 |

**Win counts:**

| comparison | wins / n |
|---|---:|
| alt_attn_stu vs alt_attn_mamba2 | 19/29 (ties 1) |
| alt_attn_stu vs pure attn | 14/29 |
| alt_attn_stu vs pure stu | 26/29 |
| alt_attn_stu vs lstm | 18/29 |

## 6. Takeaway

- At 1m, alt_attn_stu beats pure attn by +0.064 on the 62-task intersection (wins 32/62 pairwise). Layer-wise attn+X does yield a recipe that beats pure attn on mech-bench, and STU is a viable alternating branch.
- alt_attn_stu underperforms alt_attn_mamba (Mamba-1) by -0.029 mean (wins 19/62). The layer-wise recipe's benefit at 1m is specific to Mamba-1, not generic across branches.
- Adding alternating attention layers to STU is helpful: Δmean +0.318, wins 51/62 vs pure stu.
- At 10m, alt_attn_stu vs pure attn: Δmean +0.038, wins 14/29.
- At 10m, alt_attn_stu vs alt_attn_mamba2: Δmean +0.028, wins 19/29.
- Standout STU-helps tasks (vs alt_attn_mamba): `set_intersection_count` (+0.537). Standout STU-hurts tasks: `grid_two_coord` (-0.941).

CSVs: `/home/tt6444/mechanistic-bench/figures/alt_attn_stu_1m.csv`, `/home/tt6444/mechanistic-bench/figures/alt_attn_stu_10m.csv`.
