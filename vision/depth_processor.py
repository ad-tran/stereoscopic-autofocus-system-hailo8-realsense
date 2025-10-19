from utils.config import (
    TRUE_DISTANCES_M,
    INSIDE_GOOD_LIGHTING,
    INSIDE_BAD_LIGHTING,
    OUTSIDE_GOOD_LIGHTING,
    OUTSIDE_BAD_LIGHTING,
)


def correct_distance(measured_m: float, lighting_condition: list[float]) -> float:
    import bisect
    pos = bisect.bisect_left(lighting_condition, measured_m)
    if pos == 0:
        return measured_m
    elif pos == len(lighting_condition):
        return 10.0
    else:
        prev_meas = lighting_condition[pos - 1]
        next_meas = lighting_condition[pos]
        prev_true = TRUE_DISTANCES_M[pos - 1]
        next_true = TRUE_DISTANCES_M[pos]
        alpha = (measured_m - prev_meas) / (next_meas - prev_meas)
        return float(prev_true + alpha * (next_true - prev_true))


def get_lighting_lut(lichtbedingung: str | None) -> list[float]:
    if lichtbedingung == "Drinnen - Gutes Licht":
        return INSIDE_GOOD_LIGHTING
    elif lichtbedingung == "Drinnen - Schlechtes Licht":
        return INSIDE_BAD_LIGHTING
    elif lichtbedingung == "Draußen - Gutes Licht":
        return OUTSIDE_GOOD_LIGHTING
    elif lichtbedingung == "Draußen - Schlechtes Licht":
        return OUTSIDE_BAD_LIGHTING
    return INSIDE_BAD_LIGHTING