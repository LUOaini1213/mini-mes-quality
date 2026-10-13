# SOP-RFL-003 Reflow Soldering and Profile Control

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: D (synthetic) | Applies to: RFL-01, RFL-02 | Route step: 30 REFLOW

## 1 Purpose
This procedure defines the reflow profile limits for the lead-free paste used on the line and how the profile
is verified.

## 2 Profile limits
2.1 Peak temperature measured on the board: 235 to 245 degC.
2.2 Time above liquidus (217 degC): 45 to 75 seconds.
2.3 Ramp rate during preheat: no more than 3 degC per second.
2.4 Soak zone: 150 to 190 degC for 60 to 120 seconds.

## 3 Profile verification
3.1 The profile is verified once per day, at the start of shift A, with the thermocouple profiler board.
3.2 The profile is also verified after any oven maintenance and after any recipe change.
3.3 If a verified profile is outside the limits in section 2, the oven is put on HOLD in the MES and no lot may
be tracked in until the process engineer releases it.

## 4 Nitrogen atmosphere
4.1 The oxygen level in the reflow tunnel must stay below 1000 ppm when nitrogen mode is used.
4.2 An oxygen alarm stops the conveyor entry; boards already in the oven complete the cycle.

## 5 Conveyor and loading
5.1 Leave a gap of at least one board length between consecutive boards on the conveyor.
5.2 Boards from different lots may not be in the oven at the same time unless the lots belong to the same work
order.
