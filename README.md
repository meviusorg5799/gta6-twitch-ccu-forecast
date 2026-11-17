# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meviusorg5799/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meviusorg5799/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the make.fun bounty
["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
(`gta6-twitch-ccu-forecast-01`, $250 USDC, top 100 pay). `main.py`
(Python 3.13, standard library only) reads one case file at `/input`
and writes `/output/<id>.out`: for each of hours 2–24 of the release
day, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's Twitch
concurrent viewers in the Grand Theft Auto VI category, anchored on
the known hour-1 print.

## Method

A launch's release-day shape depends strongly on how big its hour-1
print already is: a launch whose category was already huge at 00:00 UTC
stays near that level all day, while one that was still tiny at 00:00
UTC (its unlock still hours away in the big regions) explodes later,
by 100x or more. In log space, the ratio of each later hour to hour 1
falls almost linearly with log(hour-1 print), but the relation flattens
at the top end, so a single global straight line under-predicts the
biggest launches.

For each hour h = 2..24 the program fits a kernel-weighted local linear
regression of log(viewers_h / viewers_1) on log(viewers_1) over the
comparable past launches the case ships (Gaussian kernel, bandwidth 1.5
in log units, centred on this case's own log hour-1 print, clamped to
the comparables' hour-1 range so a record launch extrapolates flat,
the way ELDEN RING and Cyberpunk 2077 did). The median is the fit at
the case's hour-1 print; the spread is the weighted interquartile range
of the fit's residuals, floored at 0.15 and capped at 0.5 in log units,
so the 5%–95% band stays honest without being blown out by the few
explosive single-unlock launches. Quantiles are
p1 * exp(pred + z_tau * sigma), z = (-1.645, -0.674, 0, 0.674, 1.645).
If a case shipped too few usable comparables, it falls back to the
persistence forecast (every hour at hour 1's print, spread widening
like a random walk), so the entry always produces a valid forecast.

Practice score (scored locally against SullyGnome's published hourly
averages for the four practice launches, the same source the kit
names; the local scorer was validated by reproducing the house
baseline's 0.387 exactly): **0.176 mean pinball loss** — p00 0.155,
p01 0.214, p02 0.144, p03 0.191 — against the house baseline's 0.387.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, offline, deterministic: two runs
on the same case write byte-identical output.
