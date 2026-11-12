"""Source-backed OPRLM compositions, separate from native PPM3 profiles."""

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
import math


@lru_cache(maxsize=1)
def _catalogue():
    return json.loads(files("crimm.Data").joinpath("oprlm_lipids.json").read_text())


def list_membrane_presets():
    """Return the 18 biological IDs and five PC/cholesterol family IDs."""
    data = _catalogue()
    return tuple(record["id"] for record in data["membranes"]) + tuple(data["pc_cholesterol_families"])


def get_membrane_preset(name, cholesterol_percent=None):
    """Return an independent recipe, with molecular ratios rather than mass fractions.

    Biological recipes retain both leaflets and their reference temperature.
    PC families require an explicit cholesterol percentage. Their temperature
    and physical phase remain the caller's choice.
    """
    data = _catalogue()
    if name in data["pc_cholesterol_families"]:
        try:
            fraction = float(cholesterol_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError("PC/cholesterol presets require cholesterol_percent.") from exc
        if not math.isfinite(fraction) or not 0 <= fraction < 100:
            raise ValueError("cholesterol_percent must be finite and in [0, 100).")
        ratios = {name: 100 - fraction}
        if fraction:
            ratios["CHL1"] = fraction
        record = {
            "id": name, "name": f"{name}/cholesterol", "kind": "pc_cholesterol",
            "upper_ratios": dict(ratios), "lower_ratios": dict(ratios),
            "cholesterol_percent": fraction, "temperature_K": None,
            "ppm_membrane": "MPC" if name == "DMPC" else "OPC",
            "ppm_code_approximation": name not in {"DMPC", "DOPC"},
        }
    else:
        if cholesterol_percent is not None:
            raise ValueError("Biological preset compositions do not accept a cholesterol override.")
        record = next((deepcopy(item) for item in data["membranes"] if item["id"] == name), None)
        if record is None:
            raise ValueError(f"Unknown membrane preset {name!r}. Choose one of: {', '.join(list_membrane_presets())}.")
    record.update(paper_doi=data["paper_doi"], recipe_doi=data["recipe_doi"],
                  recipe_sha256=data["recipe_sha256"], positioning_profile="DOPC")
    return record
