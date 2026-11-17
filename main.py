"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (derived from this kit's own practice data, nothing else):

  Every case ships 38-42 comparable past launches with full hourly
  Twitch curves.  A launch's release-day shape depends strongly on how
  big its hour-1 print already is: a launch whose category was already
  huge at 00:00 UTC stays near that level all day, while one that was
  still tiny at 00:00 UTC (its unlock still hours away in the big
  regions) explodes later, by 100x or more.  In log space the ratio of
  each later hour to hour 1 falls almost linearly with log(hour-1
  print), but the relation flattens at the top end, so a single global
  straight line under-predicts the biggest launches.

  For each hour h = 2..24 we therefore fit a kernel-weighted local
  linear regression of log(viewers_h / viewers_1) on log(viewers_1)
  over the comparables (Gaussian kernel, bandwidth 1.5 in log units,
  centred on this case's own log hour-1 print, clamped to the
  comparables' hour-1 range so a record launch extrapolates flat,
  like ELDEN RING / Cyberpunk 2077 did).  When the kernel's
  effective sample size at the centre is thin (below 20 launches),
  the local slope is shrunk toward 0 -- the pooled shape -- in
  proportion to the shortfall, anchored at the weighted mean, so a
  record or tiny launch does not inherit a slope fitted on a
  handful of comparables.  The median is the fit at
  the case's hour-1 print; the spread is the weighted interquartile
  range of the fit's residuals, floored at 0.15 and capped at 0.5 in
  log units, so the 5%-95% band stays honest without being blown out
  by the few explosive single-unlock launches.  Quantiles are
  p1 * exp(pred + z_tau * sigma), z = (-1.645, -0.674, 0, 0.674, 1.645).

  If a case shipped too few usable comparables, we fall back to the
  persistence forecast (every hour at hour 1's print, spread widening
  like a random walk), so the entry always produces a valid forecast.

Practice (scored locally against SullyGnome's published hourly
averages for the four practice launches, the same source the kit
names): mean pinball loss 0.176, per case p00 0.155, p01 0.214,
p02 0.144, p03 0.191, against the house baseline's 0.387.

Standard library only, offline, deterministic: no randomness, no
clock, no network; the same case file always yields the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Bandwidth of the Gaussian kernel over log(hour-1 viewers).
BANDWIDTH = 1.5
# Log-spread of an hour's forecast, floored and capped.
SIGMA_FLOOR = 0.15
SIGMA_CAP = 0.5
# A comparable counts when its release day has an hour-1 print and
# at least this many of the 24 hours present.
MIN_DAY_HOURS = 20
# Below this many usable comparables, fall back to persistence.
MIN_COMPARABLES = 8
# Persistence fallback spread, as in the house baseline.
FALLBACK_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def comparable_curves(case: dict) -> list[dict]:
    """Each comparable's release-day hourly viewers, hours 1..24.

    Skips the pre-release GTA VI comparable (no release day yet) and
    any launch whose day is too sparse to normalise.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day0 = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours: dict[int, float] = {}
        for point in comp.get("hourly") or []:
            try:
                when = parse_time(point["from"])
                viewers = point["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if not viewers or viewers <= 0:
                continue
            hour = int((when - day0).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                hours[hour] = float(viewers)
        if 1 in hours and len(hours) >= MIN_DAY_HOURS:
            curves.append(hours)
    return curves


def weighted_quantile(pairs: list[tuple[float, float]], q: float) -> float:
    """The q-quantile of values weighted by their weights.

    `pairs` is a list of (value, weight); the list is sorted by value
    first, so the result is deterministic for a given input.
    """
    ordered = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= q * total:
            return value
    return ordered[-1][0]


def local_fit(points: list[tuple[float, float]], centre: float) -> tuple[float, float]:
    """Kernel-weighted local linear fit at `centre`.

    `points` are (log hour-1 viewers, log ratio to hour 1).  Returns
    (predicted log ratio at `centre`, log spread from the weighted
    interquartile range of the residuals).
    """
    weights = [math.exp(-0.5 * ((x - centre) / BANDWIDTH) ** 2) for x, _ in points]
    total = sum(weights)
    mean_x = sum(w * x for w, (x, _) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_, y) in zip(weights, points)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, (x, _) in zip(weights, points))
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, (x, y) in zip(weights, points))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    # Shrink the local slope toward 0 (the pooled shape) when the
    # kernel's effective sample size at the centre is thin: a slope
    # fitted on a handful of launches is noise, not signal.  The
    # shrink is anchored at the weighted mean, so the fit still
    # passes through (mean_x, mean_y); at ESS >= 20 it is inactive.
    ess = total * total / sum(w * w for w in weights)
    shrink = min(1.0, ess / 20.0)
    slope *= shrink
    intercept = mean_y - slope * mean_x
    predicted = intercept + slope * centre
    residuals = [(y - (intercept + slope * x), w) for w, (x, y) in zip(weights, points)]
    q25 = weighted_quantile(residuals, 0.25)
    q75 = weighted_quantile(residuals, 0.75)
    sigma = (q75 - q25) / 1.349
    return predicted, min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)


def fallback_quantiles(p1: float, hour: int) -> list[float | int]:
    """Persistence: every hour at hour 1's print, widening spread."""
    return [
        round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)))
        for z in Z
    ]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    curves = comparable_curves(case)
    log_p1 = math.log(p1)
    usable = len(curves) >= MIN_COMPARABLES

    log_firsts = [math.log(hours[1]) for hours in curves] if usable else []
    if usable:
        # Clamp to the comparables' hour-1 range: a launch bigger than
        # every comparable takes the biggest launches' shape instead
        # of extrapolating a straight line past them.
        centre = min(max(log_p1, min(log_firsts)), max(log_firsts))
    else:
        centre = log_p1

    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        if not usable:
            quantiles = fallback_quantiles(p1, hour)
        else:
            points = [
                (math.log(hours[1]), math.log(hours[hour] / hours[1]))
                for hours in curves
                if hour in hours
            ]
            predicted, sigma = local_fit(points, centre)
            quantiles = [
                round2(p1 * math.exp(predicted + z * sigma)) for z in Z
            ]
        # Guard the contract: positive, at most 1e9, non-decreasing.
        quantiles = [min(max(float(q), 0.01), 1e9) for q in quantiles]
        for i in range(1, 5):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        quantiles = [
            int(q) if float(q).is_integer() else round(q, 2) for q in quantiles
        ]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
