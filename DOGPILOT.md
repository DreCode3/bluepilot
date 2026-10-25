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
| Low-speed curve anticipation in angle mode (`lateral_angle_ext.py`, `_DP_LEAD_*`; param `DogPilotLowSpeedLead`, default on) | Between ~20 and ~35 mph, looks ~0.9 s further ahead in the model's plan and weights it 0.8 instead of 0.5, so curves are commanded earlier. Tapers to stock below 12 mph and by 55 mph; identical to stock at and above 55 mph. | Experiment |
| Camera offset in the stock model runner (`selfdrive/modeld/dogpilot_offset.py`, param `CameraOffset`, default 0) | sunnypilot's lateral camera offset, applied in the model frame so it is the same at every distance. Corrects a lateral bias without changing the driving model. | Experiment, starts at 0 |
| Curve centering in the stock model runner (`dogpilot_offset.py` `DPCurveCentering`, param `DogPilotCurveCentering`, strength 0–2, default 0 = off) | In curves the model plans a line toward the inside, roughly in proportion to lateral acceleration. Shifts the model's view toward the outside by 0.2 m per m/s² above 0.25 m/s² (at strength 1), capped at 0.35 m and 0.2 m/s, so the model's own plan sits closer to the lane centre. Fades out between 1.8 and 2.5 m/s² (very tight curves, where the car runs short of steering and the inside line needs less of it). Off below ~20 mph, when not engaged, and during lane changes. | Experiment, starts off |
| Tight-curve boost in angle mode (`lateral_angle_ext.py` `_DP_BOOST_*`, param `DogPilotCurveBoost`, 1.0–1.5, default 1.0 = off) | Above ~2 m/s² the car delivers only 0.65–0.85 of the path angle it is sent. Scales the path angle up only for large requests: ×1 below 1.8 m/s², rising to the param's value at 2.5 m/s², faded in from 40 to 50 mph. Applied after the deviation clip, so the shadow curvature ford.h checks is unchanged; the path-angle range and rate limits still apply. | Experiment, starts off |
| hardwared stall diagnostics (`system/hardware/hardwared.py`) | Dumps thread stacks on a > 3 s main-loop stall and logs slow network/modem queries, to find the cause of a `commIssue` (deviceState went missing for 8.4 s at speed). | Diagnostics |
| DogPilot items in BluePilot > lat. tuning (comma 4) | On/off for the low-speed lead, a camera-offset slider, a curve-centering strength and a tight-curve boost, for A/B drives. | Tooling |

Settings-only experiments (BluePilot menu values) are tracked in the owner's DogPilot project,
not here.

## Base

Upstream: `BluePilotDev/bluepilot` `bp-dev` at `d27bc20a7c13eed7ec1947d4c59b98835facea87`.
