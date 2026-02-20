"""
Bolt selector for ISO metric coarse threads (M5–M24)

What it checks:
- Bolt tensile capacity against axial tensile force
- Bolt shear capacity against transverse shear force
- Internal thread shear-out for specified engagement length and nut material
- Prescribed tightening torque to achieve target preload

Key assumptions (adjust to suit your standards):
- Tensile capacity uses bolt proof stress for grade (Sp), not ultimate.
- Shear capacity uses 0.58 * Sp on stress area At (von Mises simplification).
- Internal thread shear area uses pitch diameter with 60° thread form factor ~0.5.
- Target preload F_preload = preload_factor * At * Sp (default 0.75 of proof load).
- Torque T = k * D * F_preload (k ~ 0.2 for lightly oiled steel).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class BoltSpec:
    size: str          # e.g., "M10"
    d: float           # nominal diameter (mm)
    pitch: float       # thread pitch (mm)
    At: float          # stress area (mm^2), ISO 898-1 tables (coarse)
    d2: float          # pitch diameter (mm) ~ d - 0.64952*p
    d3: float          # minor diameter internal (mm) ~ d - 1.22688*p
    hole_size: float          # 

# ISO metric coarse thread reference data (rounded typical values)
# At values from ISO 898-1 and common charts; d2/d3 via ISO formulae
def iso_metric_coarse_db() -> Dict[str, BoltSpec]:
    sizes = [
        ("M5",  5.0,  0.8),
        ("M6",  6.0,  1.0),
        ("M8",  8.0,  1.25),
        ("M10", 10.0, 1.5),
        ("M12", 12.0, 1.75),
        ("M16", 16.0, 2.0),
        ("M20", 20.0, 2.5),
        ("M24", 24.0, 3.0),
    ]
    # Stress area approximations for coarse threads (mm^2)
    At_map = {
        "M5": 14.2, "M6": 20.1, "M8": 36.6, "M10": 58.0,
        "M12": 84.3, "M16": 157.0, "M20": 245.0, "M24": 353.0
    }
    Hole_map = {
        "M3": 3.4, "M4": 4.5, "M5": 5.5, "M6": 6.6, "M8": 9.0, "M10": 11.0,
        "M12": 14.0, "M16": 18.0
    }
    db = {}
    for name, d, p in sizes:
        d2 = d - 0.64952 * p
        d3 = d - 1.22688 * p  # internal minor diameter
        db[name] = BoltSpec(
            size=name, d=d, pitch=p, At=At_map[name], d2=d2, d3=d3, hole_size=Hole_map.get(name, 20.0)
        )
    return db

# Proof stress (MPa) typical for ISO 898-1 bolt grades
PROOF_STRESS = {
    "4.6": 225.0,
    "5.6": 310.0,
    "8.8": 640.0,
    "9.8": 720.0,
    "10.9": 830.0,
    "12.9": 970.0,
}

@dataclass
class DesignInputs:
    tensile_force_N: float         # axial tensile force (N)
    shear_force_N: float           # transverse shear force (N)
    bolt_grade: str                # e.g., "8.8"
    nut_material_allow_shear_MPa: float  # allowable shear for internal thread/nut (MPa)
    engagement_length_mm: float    # thread engagement length in nut (mm)
    preload_factor: float = 0.75   # fraction of proof load for preload
    torque_k_factor: float = 0.2   # torque coefficient k (typ. 0.18–0.25, condition dependent)

@dataclass
class CheckResult:
    size: str
    hole_size: float
    ok_tension: bool
    ok_shear: bool
    ok_thread_shearout: bool
    tension_capacity_N: float
    shear_capacity_N: float
    thread_shear_capacity_N: float
    recommended_torque_Nm: float
    preload_N: float

def tensile_capacity_N(At_mm2: float, proof_stress_MPa: float) -> float:
    # Capacity based on proof load (N) = At * Sp (MPa) with MPa = N/mm^2
    return At_mm2 * proof_stress_MPa

def shear_capacity_N(At_mm2: float, proof_stress_MPa: float) -> float:
    # Simplified shear capacity using 0.58 * Sp on stress area (von Mises for ductile materials)
    return At_mm2 * (0.58 * proof_stress_MPa)

def thread_shearout_capacity_N(d2_mm: float, engagement_mm: float, allow_shear_MPa: float) -> float:
    """
    Internal thread shear-out capacity:
    A_shear ≈ π * d2 * engagement * 0.5 (60° thread form factor)
    Capacity (N) = A_shear (mm^2) * allowable shear (MPa)
    """
    A_shear_mm2 = 3.141592653589793 * d2_mm * engagement_mm * 0.5
    return A_shear_mm2 * allow_shear_MPa

def recommended_torque_Nm(k_factor: float, nominal_d_mm: float, preload_N: float) -> float:
    # T (N·m) = k * D(m) * F(N)
    return k_factor * (nominal_d_mm / 1000.0) * preload_N

def compute_preload_N(At_mm2: float, proof_stress_MPa: float, preload_factor: float) -> float:
    return preload_factor * At_mm2 * proof_stress_MPa

def select_bolt(inputs: DesignInputs,
                min_size: str = "M5",
                max_size: str = "M24") -> Optional[CheckResult]:
    db = iso_metric_coarse_db()
    sizes_order = ["M5", "M6", "M8", "M10", "M12", "M16", "M20", "M24"]

    # Filter size range
    start_idx = sizes_order.index(min_size)
    end_idx = sizes_order.index(max_size)
    candidates = sizes_order[start_idx:end_idx + 1]

    if inputs.bolt_grade not in PROOF_STRESS:
        raise ValueError(f"Unsupported bolt grade: {inputs.bolt_grade}")

    Sp = PROOF_STRESS[inputs.bolt_grade]  # MPa

    for size in candidates:
        spec = db[size]

        tension_cap = tensile_capacity_N(spec.At, Sp)
        shear_cap = shear_capacity_N(spec.At, Sp)
        thread_cap = thread_shearout_capacity_N(spec.d2, inputs.engagement_length_mm,
                                                inputs.nut_material_allow_shear_MPa)
        ok_tension = inputs.tensile_force_N <= tension_cap
        ok_shear = inputs.shear_force_N <= shear_cap
        ok_thread = inputs.tensile_force_N <= thread_cap  # axial load tends to strip threads

        if ok_tension and ok_shear and ok_thread:
            preload = compute_preload_N(spec.At, Sp, inputs.preload_factor)
            torque = recommended_torque_Nm(inputs.torque_k_factor, spec.d, preload)
            return CheckResult(
                size=size,
                hole_size=db[size].hole_size,
                ok_tension=ok_tension,
                ok_shear=ok_shear,
                ok_thread_shearout=ok_thread,
                tension_capacity_N=tension_cap,
                shear_capacity_N=shear_cap,
                thread_shear_capacity_N=thread_cap,
                recommended_torque_Nm=torque,
                preload_N=preload
            )

    return None

def summarize_result(result: Optional[CheckResult], inputs: DesignInputs) -> str:
    if result is None:
        return (
            "No bolt size within the specified range meets all criteria.\n"
            f"- Tensile demand: {inputs.tensile_force_N:,.0f} N\n"
            f"- Shear demand:   {inputs.shear_force_N:,.0f} N\n"
            f"Consider increasing size range, bolt grade, engagement length, or nut material shear strength."
        )
    return (
        f"Recommended bolt size: {result.size}\n"
        f"- Tension OK: {result.ok_tension} (capacity {result.tension_capacity_N:,.0f} N)\n"
        f"- Shear OK:   {result.ok_shear} (capacity {result.shear_capacity_N:,.0f} N)\n"
        f"- Thread shear-out OK: {result.ok_thread_shearout} (capacity {result.thread_shear_capacity_N:,.0f} N)\n"
        f"- Target preload: {result.preload_N:,.0f} N\n"
        f"- Tightening torque: {result.recommended_torque_Nm:.1f} N·m\n"
    )

def select_bolt_gh(tensile_force_N, shear_force_N):
    inputs = DesignInputs(
        tensile_force_N=tensile_force_N, # Load input from results
        shear_force_N=shear_force_N, #Load input from results
        bolt_grade="8.8",
        nut_material_allow_shear_MPa=220.0,  # mild steel nut or parent material
        engagement_length_mm=10.0, # thickness of base plate material or nut-length
        preload_factor=0.75,
        torque_k_factor=0.2
    )

    result = select_bolt(inputs, min_size="M6", max_size="M24")
    return result, summarize_result(result, inputs)

# Example usage
if __name__ == "__main__":
    inputs = DesignInputs(
        tensile_force_N=25000.0, # Load input from results
        shear_force_N=8000.0, #Load input from results
        bolt_grade="8.8",
        nut_material_allow_shear_MPa=220.0,  # mild steel nut or parent material
        engagement_length_mm=10.0, # thickness of base plate material or nut-length
        preload_factor=0.75,
        torque_k_factor=0.2
    )

    result = select_bolt(inputs, min_size="M6", max_size="M24")
    print(summarize_result(result, inputs))