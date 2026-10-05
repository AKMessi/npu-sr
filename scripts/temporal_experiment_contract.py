"""Explicit script-only static contracts; no change to production preset defaults."""

from dataclasses import replace

from npu_sr.model import MODELS, ModelSpec


def register_fused(geometry: str, state_features: bool) -> str:
    identifier = "quicksrnet-temporal-feature-experiment"
    if state_features:
        identifier = "quicksrnet-temporal-state-experiment"
        spec = ModelSpec(
            identifier,
            "QuickSRNet+RecurrentTemporalResidual",
            halo=7,
            input_channels=6,
            license="BSD-3-Clause AND MIT",
        )
    else:
        spec = ModelSpec(
            identifier,
            "QuickSRNet+TemporalResidual",
            halo=7,
            input_channels=2,
            license="BSD-3-Clause AND MIT",
        )
    if geometry == "320x270":
        spec = replace(spec, core=320, core_height=270)
    elif geometry != "256":
        raise ValueError("Unsupported static temporal geometry")
    MODELS[identifier] = spec
    return identifier
