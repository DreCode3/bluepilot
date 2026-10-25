# DogPilot — personal development branch

> **Please don't install this on your vehicle.**
>
> This branch is active, in-progress development for one specific car and its owner's own
> testing. It is not a release, has not been validated for anyone else's vehicle, and may change
> or break at any time without notice. If you're looking for BluePilot, please use the official
> project at [BluePilotDev/bluepilot](https://github.com/BluePilotDev/bluepilot) — that's where
> releases, support and community testing live.

## What this is

The `dogpilot` branch is BluePilot `bp-dev` plus a small number of experimental changes being
tried on the owner's car. Every change is a separate, clearly described commit on top of the
upstream base, so it can be reviewed or reverted on its own.

## Changes on top of BluePilot

| Change | Why | Status |
| --- | --- | --- |
| Driver-monitoring phone threshold 0.5 → 0.9 (`selfdrive/monitoring/policy.py`) | Reduces false phone-distraction alerts for the owner. Face, gaze and eye monitoring are unchanged. | Temporary — will be reverted |
| Low-speed curve anticipation in angle mode (`lateral_angle_ext.py`, `_DP_LEAD_*`; param `DogPilotLowSpeedLead`, default off) | Between ~20 and ~35 mph, looks ~0.9 s further ahead in the model's plan and weights it 0.8 instead of 0.5, so curves are commanded earlier. Tapers to stock below 12 mph and by 55 mph; identical to stock at and above 55 mph. | Experiment |
| Camera offset in the stock model runner (`selfdrive/modeld/dogpilot_offset.py`, param `CameraOffset`, default 0) | sunnypilot's lateral camera offset, applied in the model frame so it is the same at every distance. Corrects a lateral bias without changing the driving model. | Experiment, starts at 0 |
| Curve centering in the stock model runner (`dogpilot_offset.py` `DPCurveCentering`, param `DogPilotCurveCentering`, strength 0–2, default 0 = off) | In curves the model plans a line toward the inside, roughly in proportion to lateral acceleration. Shifts the model's view toward the outside by 0.2 m per m/s² above 0.25 m/s² (at strength 1), capped at 0.35 m and 0.2 m/s, so the model's own plan sits closer to the lane centre. Fades out only in extreme curves (3.0–3.8 m/s²; it used to start at 1.8, but removing the shift at the entry of 2.4–2.8 m/s² highway curves made the car cut 0.5–0.95 m, and the tight-curve boost now covers steering shortfall); the fade is held while the curve stays above 1.5 m/s² so it can't flicker. The shift is held unchanged during lane changes (the model reacts to it moving). An "early" variant (param `DogPilotCurveEarly`, off, not in the menu) also looks 27–61 m ahead and releases slowly; replay showed little benefit. Off below ~20 mph, when not engaged, and during lane changes. | Experiment, starts off |
| Tight-curve boost in angle mode (`lateral_angle_ext.py` `_DP_BOOST_*`, param `DogPilotCurveBoost`, 1.0–1.5, default 1.0 = off) | Above ~2 m/s² the car delivers only 0.65–0.85 of the path angle it is sent. Scales the path angle up only for large requests: ×1 below 1.8 m/s², rising to the param's value at 2.5 m/s², faded in from 40 to 50 mph. The level rises in 0.5 s, holds through dips while the request stays above 0.8 m/s², and falls quickly after the curve: following the request frame by frame made it switch on and off every ~3 s in a long tight curve. Applied after the deviation clip, so the shadow curvature ford.h checks is unchanged; the path-angle range and rate limits still apply. | Experiment, starts off |
| Lane changes use stock low-speed gain (`lateral_angle_ext.py` `_DP_LC_*`) | With Low Speed Adjustment Factor 1.5, a 30 mph lane change got ~1.9x the model's request, overshot and swung back. During a lane change the factor is treated as at most 1.0 (easing in over 0.5 s, out over 1.5 s). Curves and highway lane changes are unchanged. | Fix |
| Thermal reads off the deviceState path (`system/hardware/hardwared.py` `DPThermalCache`) | A sysfs thermal-zone read occasionally blocks 6-8 s; inside hardware_thread that stopped deviceState and selfdrived raised `commIssue` ("TAKE CONTROL IMMEDIATELY"). Zones are now read on their own thread and deviceState uses the latest values; slow zones are logged by name. | Fix |
| deviceState keep-alive (`system/hardware/hardwared.py` `DPDeviceStateKeepAlive`) | The sysfs stalls hit every hardware read at once (power draw too, not just thermals), so hardware_thread can still block 5-20 s. If no real deviceState has gone out for 0.75 s, the last one is republished at 2 Hz for at most 30 s per stall; a really dead hardwared still alerts. Each stall is logged with its length. | Fix |
| CPU limit restored on going onroad (`system/hardware/tici/hardware.py` `set_power_save`) | AGNOS's boot services race: comma.sh caps the CPU clusters at 1.69 GHz, init.qcom.sh sets 2.65 GHz, last one wins. On a 1.69 GHz boot CPU 4 (card, selfdrived, controlsd) saturates and engagement is refused (commIssue, selfdrivedLagging). Writes 2.65 GHz (init.qcom.sh's value) once on going onroad; thermal throttling still applies below it. | Fix |
| hardwared stall diagnostics (`system/hardware/hardwared.py`) | Dumps thread stacks on a > 3 s main-loop stall and logs slow network/modem queries, to find the cause of a `commIssue` (deviceState went missing for 8.4 s at speed). | Diagnostics |
| DogPilot items in BluePilot > lat. tuning (comma 4) | On/off for the low-speed lead, a camera-offset slider, a curve-centering strength and a tight-curve boost, for A/B drives. | Tooling |

Settings-only experiments (BluePilot menu values) are tracked in the owner's DogPilot project,
not here.

## Base

Upstream: `BluePilotDev/bluepilot` `bp-dev` at `d27bc20a7c13eed7ec1947d4c59b98835facea87`.
