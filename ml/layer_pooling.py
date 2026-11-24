"""WavLM layer-list pooling helpers (Goal 2).

``ml/train_cm.py --merge-layers N`` merges a prefix of hidden states,
``hidden_states[0:N]``. That cannot express the L6 to L11 pooling the
Hyper Research report recommends, and ``--merge-layers 12`` is not L6
to L11 pooling: hidden_states[0] is the embedding output, so the prefix
never even reads transformer layer 11.

A layer list is written in transformer-layer numbering, 0 to 11 for
WavLM-base-plus, the same numbering the report uses. It maps to hidden
states by adding one: transformer layer ``k`` is ``hidden_states[k+1]``.
So ``--merge-layer-list 6,7,8,9,10,11`` pools transformer layers 6 to
11, i.e. hidden states 7 to 12.

This module has no torch dependency, so the parsing and validation can
be unit tested on a machine that cannot run training.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

N_ENCODER_LAYERS = 12


def parse_merge_layer_list(spec: Optional[str]) -> Optional[List[int]]:
    """Parse ``"6,7,8,9,10,11"`` into ``[6, 7, 8, 9, 10, 11]``.

    Returns ``None`` for an empty or missing spec. Raises ``ValueError``
    for a non-integer entry, an out-of-range layer, or a duplicate.
    """
    if spec is None:
        return None
    items = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not items:
        return None
    layers: List[int] = []
    for item in items:
        try:
            layer = int(item)
        except ValueError:
            raise ValueError(
                f"--merge-layer-list entry {item!r} is not an integer; "
                "expected comma-separated transformer layer indices 0-11"
            ) from None
        if layer < 0 or layer >= N_ENCODER_LAYERS:
            raise ValueError(
                f"--merge-layer-list layer {layer} is out of range; "
                f"WavLM-base-plus has transformer layers 0-{N_ENCODER_LAYERS - 1}"
            )
        if layer in layers:
            raise ValueError(
                f"--merge-layer-list repeats layer {layer}; each layer once"
            )
        layers.append(layer)
    return layers


def hidden_state_indices(layer_list: Sequence[int]) -> List[int]:
    """Transformer layer indices -> WavLM hidden_states indices (+1)."""
    return [int(layer) + 1 for layer in layer_list]


def resolve_merge_config(
    merge_layers: Optional[int], merge_layer_list: Optional[Sequence[int]]
) -> dict:
    """Validate the two merge options and return the effective head config.

    Returns ``{"merge_layers": count or None, "merge_layer_list": list or
    None, "last_used_layer": transformer index of the highest layer the
    head reads, or None}``. ``last_used_layer`` anchors the
    ``--unfreeze-top-n`` window in ``ml/train_cm.py``. Raises
    ``ValueError`` when both merge options are given, or when
    ``merge_layers`` is out of range.
    """
    if merge_layers is not None and merge_layer_list is not None:
        raise ValueError(
            "use only one of --merge-layers (a hidden-states prefix) and "
            "--merge-layer-list (an explicit transformer layer list)"
        )
    if merge_layer_list is not None:
        layers = list(merge_layer_list)
        if not layers:
            raise ValueError("--merge-layer-list must name at least one layer")
        return {
            "merge_layers": len(layers),
            "merge_layer_list": layers,
            "last_used_layer": max(layers),
        }
    if merge_layers is not None:
        if merge_layers < 1 or merge_layers > N_ENCODER_LAYERS + 1:
            raise ValueError(
                f"--merge-layers must be in [1, {N_ENCODER_LAYERS + 1}], "
                f"got {merge_layers}"
            )
        return {
            "merge_layers": int(merge_layers),
            "merge_layer_list": None,
            "last_used_layer": merge_layers - 2,
        }
    return {"merge_layers": None, "merge_layer_list": None, "last_used_layer": None}
