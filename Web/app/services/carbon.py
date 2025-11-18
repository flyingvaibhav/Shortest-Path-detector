"""Carbon accounting helpers for TreeSense."""
from __future__ import annotations

from typing import Iterable, Sequence

CO2_CONVERSION = 44 / 12  # kg CO2e per kg C


def _as_sequence(values: Iterable[float]) -> Sequence[float]:
    return tuple(float(v) for v in values)


def estimate_carbon_sequestration(
    tree_count: int,
    biomass_values: Sequence[float] = (50, 250, 500, 1000),
    annual_rates: Sequence[float] = (2, 10, 25),
    carbon_fraction: float = 0.5,
) -> dict[str, dict[str, Sequence[float]]]:
    """Estimate stored and annual CO2e based on tree counts."""

    safe_count = max(int(tree_count or 0), 0)
    biomass_values = _as_sequence(biomass_values)
    annual_rates = _as_sequence(annual_rates)

    stock_per_tree = []
    stock_total = []
    for biomass in biomass_values:
        carbon_kg = biomass * carbon_fraction
        co2e_per_tree = carbon_kg * CO2_CONVERSION
        stock_per_tree.append(co2e_per_tree)
        stock_total.append(co2e_per_tree * safe_count)

    annual_per_tree = []
    annual_total = []
    for rate in annual_rates:
        co2e_per_tree_year = rate * CO2_CONVERSION
        annual_per_tree.append(co2e_per_tree_year)
        annual_total.append(co2e_per_tree_year * safe_count)

    return {
        "stock": {
            "assumptions_biomass": biomass_values,
            "per_tree_CO2e_kg": tuple(stock_per_tree),
            "total_CO2e_kg": tuple(stock_total),
        },
        "annual": {
            "assumptions_annual_rate": annual_rates,
            "per_tree_CO2e_kg_per_year": tuple(annual_per_tree),
            "total_CO2e_kg_per_year": tuple(annual_total),
        },
    }


def estimate_oxygen_output(tree_count: int, age_group: str = "mature") -> dict[str, object]:
    """Estimate annual oxygen production for detected trees."""

    safe_count = max(int(tree_count or 0), 0)
    age_group = (age_group or "").lower()

    oxygen_map = {
        "young": (5, 20),
        "medium": (40, 80),
        "mature": (100, 120),
    }

    if age_group not in oxygen_map:
        raise ValueError(f"Invalid age_group '{age_group}'. Use young/medium/mature.")

    per_tree_min, per_tree_max = oxygen_map[age_group]
    total_min = safe_count * per_tree_min
    total_max = safe_count * per_tree_max
    total_avg = safe_count * ((per_tree_min + per_tree_max) / 2)

    return {
        "tree_count": safe_count,
        "age_group": age_group,
        "o2_per_tree_range": (per_tree_min, per_tree_max),
        "estimated_o2_total_range": (total_min, total_max),
        "estimated_o2_average": total_avg,
    }


def estimate_pm_capture(
    tree_count: int,
    pm_per_tree_low: float = 0.10,
    pm_per_tree_typ: float = 0.15,
    pm_per_tree_high: float = 1.0,
    use_deposition: bool = False,
    C: float | None = None,
    Vd: float | None = None,
    canopy_area: float | None = None,
) -> dict[str, object]:
    """Estimate particulate matter captured per year using heuristic or deposition model."""

    safe_count = max(int(tree_count or 0), 0)
    estimates: dict[str, object] = {
        "tree_count": safe_count,
        "simple_method": {
            "low_kg_per_year": safe_count * pm_per_tree_low,
            "typical_kg_per_year": safe_count * pm_per_tree_typ,
            "high_kg_per_year": safe_count * pm_per_tree_high,
        },
    }

    if use_deposition:
        if C is None or Vd is None or canopy_area is None:
            raise ValueError("For deposition model, provide C, Vd, and canopy_area.")

        T = 31_536_000  # seconds per year
        pm_tree_year = (Vd * C * canopy_area * T) / 1_000_000  # µg → kg
        estimates["deposition_model"] = {
            "pm_per_tree_kg_per_year": pm_tree_year,
            "pm_total_kg_per_year": pm_tree_year * safe_count,
            "inputs": {"C": C, "Vd": Vd, "canopy_area": canopy_area},
        }

    return estimates
