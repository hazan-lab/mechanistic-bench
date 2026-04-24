# Extended LM downstream evals

Checkpoints: `mechbench-{150m,50m}-{attn,mamba2,alt-attn-mamba2,headwise-mamba2}` at their final step (2862 for 150m, 954 for 50m).

Evaluator script: `scripts/eval_lm_extended.py`. Raw per-run JSONs under `results/raw/`.

Runs included: 150m-attn, 150m-mamba2, 150m-alt-attn-mamba2, 50m-attn, 50m-mamba2, 50m-alt-attn-mamba2, 50m-headwise-mamba2

## Summary (headline metric per task)

| task | metric | 150m-attn | 150m-mamba2 | 150m-alt-attn-mamba2 | 50m-attn | 50m-mamba2 | 50m-alt-attn-mamba2 | 50m-headwise-mamba2 | winner |
|---|---|---|---|---|---|---|---|---|---|
| arc_easy | eval/downstream/arc_easy_acc | 0.3649 | 0.3772 | **0.3825** | 0.3175 | 0.3000 | 0.3053 | 0.2947 | 150m-alt-attn-mamba2 |
| arc_easy_ppl | eval/downstream_ce_loss/arc_easy_ppl_ce_loss | 1.5152 | **1.4849** | 1.5360 | 1.7223 | 1.6858 | 1.7037 | 1.7189 | 150m-mamba2 |
| commonsense_qa | eval/downstream/commonsense_qa_len_norm | 0.2662 | 0.2621 | **0.2686** | 0.2383 | 0.2367 | 0.2342 | 0.2228 | 150m-alt-attn-mamba2 |
| hellaswag | eval/downstream/hellaswag_len_norm | 0.2708 | **0.2768** | 0.2744 | 0.2553 | 0.2547 | 0.2553 | 0.2539 | 150m-mamba2 |
| natural_qs_open_ppl | eval/downstream_ce_loss/natural_qs_open_ppl_ce_loss | 1.9182 | **1.8616** | 1.8860 | 2.1757 | 2.1358 | 2.1493 | 2.1462 | 150m-mamba2 |
| openbook_qa | eval/downstream/openbook_qa_len_norm | **0.2680** | 0.2580 | 0.2640 | 0.2640 | 0.2640 | 0.2660 | 0.2600 | 150m-attn |
| piqa | eval/downstream/piqa_len_norm | **0.6039** | 0.5947 | 0.5979 | 0.5359 | 0.5462 | 0.5462 | 0.5408 | 150m-attn |
| sciq | eval/downstream/sciq_acc | **0.5540** | 0.5330 | 0.5490 | 0.2560 | 0.2550 | 0.2690 | 0.2520 | 150m-attn |
| social_iqa | eval/downstream/social_iqa_len_norm | **0.4084** | 0.3982 | 0.4028 | 0.3961 | 0.3941 | 0.3925 | 0.3987 | 150m-attn |
| trivia_qa_wiki_ppl | eval/downstream_ce_loss/trivia_qa_wiki_ppl_ce_loss | 2.1869 | **2.1282** | 2.2111 | 2.5720 | 2.5519 | 2.5219 | 2.5264 | 150m-mamba2 |
| winogrande | eval/downstream/winogrande_acc | 0.5075 | 0.4980 | 0.4972 | **0.5138** | 0.5004 | 0.5036 | 0.5083 | 50m-attn |

## Per-scale win counts (headline metric)

### 150m

| arch | wins |
|---|---|
| 150m-attn | 5 |
| 150m-mamba2 | 4 |
| 150m-alt-attn-mamba2 | 2 |
| total tasks scored | 11 |

### 50m

| arch | wins |
|---|---|
| 50m-attn | 4 |
| 50m-mamba2 | 3 |
| 50m-alt-attn-mamba2 | 5 |
| 50m-headwise-mamba2 | 1 |
| total tasks scored | 11 |

## Per-task detail (all reported metric keys)

### arc_easy

| run | eval/downstream/arc_easy_acc |
|---|---|
| 150m-attn | 0.3649 |
| 150m-mamba2 | 0.3772 |
| 150m-alt-attn-mamba2 | 0.3825 |
| 50m-attn | 0.3175 |
| 50m-mamba2 | 0.3000 |
| 50m-alt-attn-mamba2 | 0.3053 |
| 50m-headwise-mamba2 | 0.2947 |

### arc_easy_ppl

| run | eval/downstream_ce_loss/arc_easy_ppl_ce_loss |
|---|---|
| 150m-attn | 1.5152 |
| 150m-mamba2 | 1.4849 |
| 150m-alt-attn-mamba2 | 1.5360 |
| 50m-attn | 1.7223 |
| 50m-mamba2 | 1.6858 |
| 50m-alt-attn-mamba2 | 1.7037 |
| 50m-headwise-mamba2 | 1.7189 |

### commonsense_qa

| run | eval/downstream/commonsense_qa_len_norm |
|---|---|
| 150m-attn | 0.2662 |
| 150m-mamba2 | 0.2621 |
| 150m-alt-attn-mamba2 | 0.2686 |
| 50m-attn | 0.2383 |
| 50m-mamba2 | 0.2367 |
| 50m-alt-attn-mamba2 | 0.2342 |
| 50m-headwise-mamba2 | 0.2228 |

### hellaswag

| run | eval/downstream/hellaswag_len_norm |
|---|---|
| 150m-attn | 0.2708 |
| 150m-mamba2 | 0.2768 |
| 150m-alt-attn-mamba2 | 0.2744 |
| 50m-attn | 0.2553 |
| 50m-mamba2 | 0.2547 |
| 50m-alt-attn-mamba2 | 0.2553 |
| 50m-headwise-mamba2 | 0.2539 |

### natural_qs_open_ppl

| run | eval/downstream_ce_loss/natural_qs_open_ppl_ce_loss |
|---|---|
| 150m-attn | 1.9182 |
| 150m-mamba2 | 1.8616 |
| 150m-alt-attn-mamba2 | 1.8860 |
| 50m-attn | 2.1757 |
| 50m-mamba2 | 2.1358 |
| 50m-alt-attn-mamba2 | 2.1493 |
| 50m-headwise-mamba2 | 2.1462 |

### openbook_qa

| run | eval/downstream/openbook_qa_len_norm |
|---|---|
| 150m-attn | 0.2680 |
| 150m-mamba2 | 0.2580 |
| 150m-alt-attn-mamba2 | 0.2640 |
| 50m-attn | 0.2640 |
| 50m-mamba2 | 0.2640 |
| 50m-alt-attn-mamba2 | 0.2660 |
| 50m-headwise-mamba2 | 0.2600 |

### piqa

| run | eval/downstream/piqa_len_norm |
|---|---|
| 150m-attn | 0.6039 |
| 150m-mamba2 | 0.5947 |
| 150m-alt-attn-mamba2 | 0.5979 |
| 50m-attn | 0.5359 |
| 50m-mamba2 | 0.5462 |
| 50m-alt-attn-mamba2 | 0.5462 |
| 50m-headwise-mamba2 | 0.5408 |

### sciq

| run | eval/downstream/sciq_acc |
|---|---|
| 150m-attn | 0.5540 |
| 150m-mamba2 | 0.5330 |
| 150m-alt-attn-mamba2 | 0.5490 |
| 50m-attn | 0.2560 |
| 50m-mamba2 | 0.2550 |
| 50m-alt-attn-mamba2 | 0.2690 |
| 50m-headwise-mamba2 | 0.2520 |

### social_iqa

| run | eval/downstream/social_iqa_len_norm |
|---|---|
| 150m-attn | 0.4084 |
| 150m-mamba2 | 0.3982 |
| 150m-alt-attn-mamba2 | 0.4028 |
| 50m-attn | 0.3961 |
| 50m-mamba2 | 0.3941 |
| 50m-alt-attn-mamba2 | 0.3925 |
| 50m-headwise-mamba2 | 0.3987 |

### trivia_qa_wiki_ppl

| run | eval/downstream_ce_loss/trivia_qa_wiki_ppl_ce_loss |
|---|---|
| 150m-attn | 2.1869 |
| 150m-mamba2 | 2.1282 |
| 150m-alt-attn-mamba2 | 2.2111 |
| 50m-attn | 2.5720 |
| 50m-mamba2 | 2.5519 |
| 50m-alt-attn-mamba2 | 2.5219 |
| 50m-headwise-mamba2 | 2.5264 |

### winogrande

| run | eval/downstream/winogrande_acc |
|---|---|
| 150m-attn | 0.5075 |
| 150m-mamba2 | 0.4980 |
| 150m-alt-attn-mamba2 | 0.4972 |
| 50m-attn | 0.5138 |
| 50m-mamba2 | 0.5004 |
| 50m-alt-attn-mamba2 | 0.5036 |
| 50m-headwise-mamba2 | 0.5083 |

