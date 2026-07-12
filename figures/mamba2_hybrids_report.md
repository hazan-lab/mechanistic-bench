# Mamba-2 hybrids vs Mamba-1 / Mamba-2 / attn baselines

Run date: 2026-04-23. Data dirs: `/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136`, `/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard`.

## 1. Summary

- **alt_attn hybrid:** on 62 tasks with both variants, mean tok_acc 0.682 (mamba-1) vs 0.618 (mamba-2), Δmean = -0.064. Mamba-2 wins on 20/62 tasks.
- **headwise hybrid:** on 62 tasks with both variants, mean tok_acc 0.495 (mamba-1) vs 0.495 (mamba-2), Δmean = -0.000. Mamba-2 wins on 31/62 tasks.
- **alt_attn extremes:** biggest regression `grid_two_coord` 1.000→0.094 (-0.906); biggest gain `dual_query_hop` 0.280→0.999 (+0.718).
- **headwise extremes:** biggest regression `counting` 0.913→0.230 (-0.683); biggest gain `dual_query_hop` 0.287→0.842 (+0.555).
- **10m absolute (mean tok_acc):** attn 0.255, alt_attn_mamba2 0.265, headwise_mamba2 0.188, lstm 0.196, stu 0.163.

## 2. 1m head-to-head: Mamba-1 vs Mamba-2 hybrids

Final `tok_acc`. Sorted by max |Δ| across the two hybrid pairs, descending.
Bold entries have |Δ| ≥ 0.1 (the new value is bolded).

| task | alt_attn_mamba_1 | alt_attn_mamba_2 | Δ alt | headwise_mamba_1 | headwise_mamba_2 | Δ hw |
|---|---:|---:|---:|---:|---:|---:|
| grid_two_coord | 1.000 | **0.094** | -0.906 | 0.024 | 0.022 | -0.002 |
| grid_retrieval | 1.000 | **0.104** | -0.896 | 0.031 | 0.020 | -0.012 |
| grid_three_coord | 1.000 | **0.158** | -0.842 | 0.029 | 0.024 | -0.004 |
| video_frame_retrieval | 1.000 | **0.237** | -0.763 | 0.033 | 0.045 | +0.012 |
| dual_query_hop | 0.280 | **0.999** | +0.718 | 0.287 | **0.842** | +0.555 |
| counting | 0.995 | **0.425** | -0.570 | 0.913 | **0.230** | -0.683 |
| video_cell_mode | 0.996 | **0.380** | -0.616 | 0.222 | **0.071** | -0.150 |
| first_vs_last | 0.967 | **0.576** | -0.391 | 0.249 | **0.415** | +0.166 |
| induction | 1.000 | 1.000 | +0.000 | 0.870 | **0.491** | -0.379 |
| dual_hop_retrieve | 0.933 | 0.985 | +0.052 | 0.470 | **0.738** | +0.269 |
| substring_locate | 0.191 | **0.366** | +0.175 | 0.144 | 0.144 | +0.000 |
| multi_induction | 1.000 | 1.000 | +0.000 | 0.879 | **0.721** | -0.158 |
| temporal_ordering | 0.483 | **0.609** | +0.126 | 0.507 | 0.483 | -0.023 |
| set_intersection_count | 0.461 | 0.447 | -0.014 | 0.106 | **0.220** | +0.113 |
| interleave | 1.000 | 1.000 | -0.000 | 0.434 | **0.332** | -0.102 |
| grid_multihop | 0.363 | 0.451 | +0.088 | 0.229 | 0.265 | +0.035 |
| noisy_copy | 0.923 | 0.839 | -0.084 | 0.026 | 0.035 | +0.009 |
| token_transition | 0.991 | 0.955 | -0.036 | 0.764 | 0.845 | +0.081 |
| state_tracking | 0.883 | 0.934 | +0.051 | 0.915 | 0.964 | +0.049 |
| batch_two_hop | 0.434 | 0.431 | -0.002 | 0.327 | 0.366 | +0.038 |
| state_retrieve | 0.998 | 0.999 | +0.001 | 0.957 | 0.991 | +0.034 |
| last_tagged | 1.000 | 1.000 | +0.000 | 0.963 | 0.992 | +0.029 |
| selective_copy | 0.675 | 0.677 | +0.002 | 0.027 | 0.056 | +0.029 |
| three_hop | 0.296 | 0.269 | -0.027 | 0.259 | 0.278 | +0.020 |
| deep_hop | 0.442 | 0.443 | +0.001 | 0.344 | 0.366 | +0.022 |
| multi_state_tracking | 0.968 | 0.970 | +0.002 | 0.973 | 0.990 | +0.018 |
| conditional_recall | 0.166 | 0.168 | +0.002 | 0.157 | 0.141 | -0.017 |
| two_hop | 0.241 | 0.230 | -0.011 | 0.221 | 0.234 | +0.014 |
| k_hop | 0.297 | 0.285 | -0.012 | 0.280 | 0.280 | +0.000 |
| union_lookup | 0.152 | 0.153 | +0.001 | 0.085 | 0.095 | +0.010 |
| nested_3_hop | 0.404 | 0.404 | +0.000 | 0.416 | 0.407 | -0.009 |
| variable_lookup | 0.213 | 0.221 | +0.008 | 0.214 | 0.219 | +0.005 |
| nearest_key | 0.141 | 0.148 | +0.008 | 0.082 | 0.088 | +0.006 |
| short_associative | 0.251 | 0.244 | -0.007 | 0.228 | 0.229 | +0.002 |
| triple_recall | 0.154 | 0.153 | -0.001 | 0.141 | 0.134 | -0.007 |
| quad_recall | 0.152 | 0.147 | -0.004 | 0.142 | 0.149 | +0.006 |
| assignment_chain | 0.460 | 0.454 | -0.006 | 0.442 | 0.446 | +0.004 |
| needle | 0.032 | 0.037 | +0.005 | 0.011 | 0.017 | +0.006 |
| associative | 0.103 | 0.101 | -0.002 | 0.052 | 0.056 | +0.004 |
| mode_tagged | 1.000 | 1.000 | +0.000 | 0.890 | 0.886 | -0.004 |
| hop_distance_bucket | 0.621 | 0.624 | +0.003 | 0.600 | 0.600 | +0.000 |
| copy_count | 0.990 | 0.992 | +0.002 | 0.990 | 0.990 | +0.000 |
| reverse_copy | 0.999 | 1.000 | +0.001 | 0.164 | 0.166 | +0.002 |
| induction_gap | 1.000 | 1.000 | +0.000 | 0.890 | 0.889 | -0.001 |
| nested_lookup | 0.343 | 0.342 | -0.001 | 0.340 | 0.339 | -0.001 |
| sort_top2 | 0.677 | 0.678 | +0.001 | 0.674 | 0.674 | +0.000 |
| selective_parity | 1.000 | 1.000 | +0.000 | 0.120 | 0.121 | +0.000 |
| compress | 1.000 | 1.000 | -0.000 | 1.000 | 1.000 | +0.000 |
| piecewise_denoise | 0.997 | 0.997 | +0.000 | 0.997 | 0.997 | +0.000 |
| sort | 1.000 | 1.000 | -0.000 | 1.000 | 0.999 | -0.000 |
| col_parity | 0.462 | 0.462 | +0.000 | 0.462 | 0.462 | +0.000 |
| copy | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| copy_offset | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| cumulative_sum | 0.119 | 0.119 | +0.000 | 0.119 | 0.119 | +0.000 |
| delayed_echo | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| longest_run | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| mode | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| parity | 0.514 | 0.514 | +0.000 | 0.514 | 0.514 | +0.000 |
| patch_match | 0.496 | 0.496 | +0.000 | 0.496 | 0.496 | +0.000 |
| pattern_completion | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| running_max | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| threshold | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| batch_recall | — | 0.157 | — | — | 0.143 | — |

## 3. 1m aggregates

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| alt_attn_mamba (m1) | 62 | 0.682 |
| alt_attn_mamba2 (m2) | 63 | 0.611 |
| headwise (m1) | 62 | 0.495 |
| headwise_mamba2 (m2) | 63 | 0.489 |
| attn | 62 | 0.588 |
| mamba | 62 | 0.381 |
| mamba2 | 29 | 0.436 |

On common-task subsets: alt-hybrid mamba-2 wins 20/62 tasks, headwise-hybrid mamba-2 wins 31/62 tasks over their mamba-1 counterparts.

## 4. 10m absolute

Final `tok_acc`. Flagged `*` when `alt_attn_mamba2` or `headwise_mamba2` beats `attn` on that task.

| task | attn_10m | alt_attn_mamba2_10m | headwise_mamba2_10m | lstm_10m | stu_10m |
|---|---:|---:|---:|---:|---:|
| assignment_chain | 0.324 | 0.280 | 0.226 | 0.297 | 0.174 |
| batch_recall * | 0.096 | 0.094 | 0.096 | 0.092 | 0.095 |
| batch_two_hop | 0.183 | 0.182 | 0.180 | 0.190 | 0.179 |
| conditional_recall | 0.113 | 0.101 | 0.052 | 0.098 | 0.047 |
| deep_hop | 0.260 | 0.211 | 0.153 | 0.268 | 0.057 |
| dual_hop_retrieve * | 0.113 | 0.126 | 0.077 | 0.091 | 0.126 |
| dual_query_hop | 0.140 | 0.124 | 0.083 | 0.143 | 0.101 |
| first_vs_last | 0.461 | 0.272 | 0.078 | 0.080 | 0.082 |
| grid_multihop | 0.430 | 0.307 | 0.312 | 0.217 | 0.168 |
| grid_three_coord * | 0.043 | 0.071 | 0.028 | 0.031 | 0.043 |
| grid_two_coord * | 0.042 | 0.061 | 0.024 | 0.033 | 0.038 |
| hop_distance_bucket | 0.408 | 0.395 | 0.395 | 0.408 | 0.236 |
| k_hop | 0.186 | 0.182 | 0.166 | 0.186 | 0.117 |
| last_tagged * | 0.168 | 0.999 | 0.562 | 0.180 | 0.160 |
| mode_tagged | 0.914 | 0.826 | 0.818 | 0.830 | 0.824 |
| nested_3_hop | 0.256 | 0.233 | 0.237 | 0.252 | 0.264 |
| nested_lookup * | 0.184 | 0.206 | 0.181 | 0.184 | 0.164 |
| quad_recall | 0.098 | 0.095 | 0.087 | 0.100 | 0.096 |
| set_intersection_count | 0.244 | 0.177 | 0.070 | 0.166 | 0.082 |
| sort_top2 * | 0.559 | 0.564 | 0.562 | 0.558 | 0.479 |
| substring_locate * | 0.062 | 0.211 | 0.074 | 0.098 | 0.102 |
| temporal_ordering | 0.574 | 0.519 | 0.491 | 0.504 | 0.504 |
| three_hop * | 0.131 | 0.160 | 0.126 | 0.119 | 0.039 |
| triple_recall * | 0.095 | 0.097 | 0.082 | 0.094 | 0.092 |
| two_hop * | 0.119 | 0.132 | 0.060 | 0.111 | 0.123 |
| union_lookup | 0.088 | 0.083 | 0.040 | 0.104 | 0.062 |
| variable_lookup * | 0.114 | 0.121 | 0.109 | 0.111 | 0.091 |
| video_cell_mode | 0.975 | 0.791 | 0.044 | 0.080 | 0.139 |
| video_frame_retrieval * | 0.025 | 0.077 | 0.034 | 0.066 | 0.045 |

## 5. 10m aggregates

| arch | n_tasks | mean tok_acc |
|---|---:|---:|
| attn | 29 | 0.255 |
| alt_attn_mamba2 | 29 | 0.265 |
| headwise_mamba2 | 29 | 0.188 |
| lstm | 29 | 0.196 |
| stu | 29 | 0.163 |

## 6. Notes

### Param counts (median per arch)

| arch | params @ 1m | params @ 10m |
|---|---:|---:|
| alt_attn_mamba | 1,010,760 | — |
| alt_attn_mamba2 | 1,199,524 | 9,731,000 |
| headwise | 961,800 | — |
| headwise_mamba2 | 1,153,444 | 8,987,320 |
| attn | 992,896 | 8,217,920 |
| mamba | 1,004,528 | — |
| mamba2 | 989,910 | — |
| lstm | — | 11,515,200 |
| stu | — | 5,821,760 |

### Anomalies / skipped runs

| store | task | arch | scale | issue |
|---|---|---|---|---|
| baselines | multi_induction | mamba | 1m | collapsed tok_acc=0.007 |

### Coverage gaps at 1m

- Tasks missing at least one of `alt_attn_mamba` or `alt_attn_mamba2`: 1 → batch_recall
- Tasks missing at least one of `headwise` or `headwise_mamba2`: 1 → batch_recall

CSVs: `/home/tt6444/mechanistic-bench/figures/mamba2_hybrids_1m.csv`, `/home/tt6444/mechanistic-bench/figures/mamba2_hybrids_10m.csv`.
