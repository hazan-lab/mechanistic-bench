"""Manual primitive-coverage taxonomy for the 27/32-task suite.

Each task is mapped to a primary primitive and any secondary primitives.
The primitives form a flat axis space so we can build a coverage matrix
showing every primitive has redundancy and no two tasks fully overlap.
"""
from __future__ import annotations

# primary primitive per task (used for drop-family ablation)
PRIMARY_PRIMITIVE = {
    # exact / point retrieval (positional copy & induction)
    "copy": "point_retrieval",
    "copy_offset": "point_retrieval",
    "reverse_copy": "point_retrieval",
    "needle": "point_retrieval",
    "induction": "point_retrieval",
    "induction_gap": "point_retrieval",
    "multi_induction": "point_retrieval",
    # content-addressable retrieval (key-value lookup)
    "associative": "content_lookup",
    "short_associative": "content_lookup",
    "batch_recall": "content_lookup",
    "conditional_recall": "content_lookup",
    "last_tagged": "content_lookup",
    "first_vs_last": "content_lookup",
    "mode_tagged": "content_lookup",
    # aggregation / reduction over a stream
    "counting": "aggregation",
    "parity": "aggregation",
    "cumulative_sum": "aggregation",
    "threshold": "aggregation",
    "mode": "aggregation",
    "longest_run": "aggregation",
    "running_max": "aggregation",
    # finite-state tracking
    "state_tracking": "finite_state",
    "multi_state_tracking": "finite_state",
    "pattern_completion": "finite_state",
    "token_transition": "finite_state",
    # gating / selective filtering
    "selective_copy": "filter",
    "selective_parity": "filter",
    "compress": "filter",
    "interleave": "filter",
    "noisy_copy": "filter",
    "sort": "filter",  # selective ordering
    # multi-hop / compositional retrieval
    "two_hop": "multi_hop",
    "three_hop": "multi_hop",
    "deep_hop": "multi_hop",
    "k_hop": "multi_hop",
    "dual_hop_retrieve": "multi_hop",
    "batch_two_hop": "multi_hop",
    "dual_query_hop": "multi_hop",
    "nested_lookup": "multi_hop",
    "nested_3_hop": "multi_hop",
    "hop_distance_bucket": "multi_hop",
    "triple_recall": "multi_hop",
    "quad_recall": "multi_hop",
    "union_lookup": "multi_hop",
    "variable_lookup": "multi_hop",
    "assignment_chain": "multi_hop",
    # 2D / structured-spatial
    "grid_retrieval": "spatial_2d",
    "col_parity": "spatial_2d",
    "patch_match": "spatial_2d",
    "grid_two_coord": "spatial_2d",
    "grid_three_coord": "spatial_2d",
    "grid_multihop": "spatial_2d",
    "sort_top2": "spatial_2d",
    "set_intersection_count": "spatial_2d",
    "temporal_ordering": "spatial_2d",
    "substring_locate": "spatial_2d",
    "video_frame_retrieval": "spatial_2d",
    "video_cell_mode": "spatial_2d",
    # continuous-valued / time-series-like
    "delayed_echo": "continuous",
    "piecewise_denoise": "continuous",
    "nearest_key": "continuous",
    # compound primitives (mixed)
    "copy_count": "compound",
    "state_retrieve": "compound",
}

# secondary primitives (which other primitives a task partially probes)
SECONDARY = {
    "selective_copy": ["filter", "point_retrieval"],
    "selective_parity": ["filter", "aggregation"],
    "interleave": ["filter", "point_retrieval"],
    "compress": ["filter", "aggregation"],
    "two_hop": ["multi_hop", "content_lookup"],
    "three_hop": ["multi_hop", "content_lookup"],
    "k_hop": ["multi_hop", "content_lookup"],
    "deep_hop": ["multi_hop", "content_lookup"],
    "grid_retrieval": ["spatial_2d", "point_retrieval"],
    "grid_two_coord": ["spatial_2d", "content_lookup"],
    "grid_three_coord": ["spatial_2d", "content_lookup"],
    "grid_multihop": ["spatial_2d", "multi_hop"],
    "col_parity": ["spatial_2d", "aggregation"],
    "patch_match": ["spatial_2d", "content_lookup"],
    "delayed_echo": ["continuous", "point_retrieval"],
    "piecewise_denoise": ["continuous", "aggregation"],
    "nearest_key": ["continuous", "content_lookup"],
    "copy_count": ["compound", "point_retrieval", "aggregation"],
    "state_retrieve": ["compound", "finite_state", "content_lookup"],
    "mode_tagged": ["content_lookup", "aggregation"],
    "induction": ["point_retrieval", "content_lookup"],
    "induction_gap": ["point_retrieval", "content_lookup"],
    "multi_induction": ["point_retrieval", "content_lookup"],
    "noisy_copy": ["filter", "point_retrieval"],
}

# Family aliases for the paper's three-level grouping:
#   Family ⊃ Primitive ⊃ Tasks
FAMILY_OF_PRIMITIVE = {
    "point_retrieval": "retrieval",
    "content_lookup": "retrieval",
    "aggregation": "aggregation",
    "finite_state": "aggregation",
    "filter": "extended",
    "multi_hop": "multi_hop",
    "spatial_2d": "multimodal",
    "continuous": "multimodal",
    "compound": "compound",
}

PRIMITIVES = sorted(set(PRIMARY_PRIMITIVE.values()))
FAMILIES = sorted(set(FAMILY_OF_PRIMITIVE.values()))

if __name__ == "__main__":
    # quick coverage report
    from collections import Counter
    c = Counter(PRIMARY_PRIMITIVE.values())
    print("Primitives and # tasks per primitive:")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v}")
    print("\nFamilies:")
    fam_counts = Counter(FAMILY_OF_PRIMITIVE[p] for p in PRIMARY_PRIMITIVE.values())
    for k, v in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v}")
