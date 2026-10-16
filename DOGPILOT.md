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

Settings-only experiments (BluePilot menu values) are tracked in the owner's DogPilot project,
not here.

## Base

Upstream: `BluePilotDev/bluepilot` `bp-dev` at `d27bc20a7c13eed7ec1947d4c59b98835facea87`.
