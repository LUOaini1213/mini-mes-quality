# SOP-PNP-002 Pick-and-Place Operation

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: B (synthetic) | Applies to: PNP-01, PNP-02, PNP-03 | Route step: 20 PLACE

## 1 Purpose
This procedure covers feeder setup, nozzle checks, first-article inspection and placement alarms on the
pick-and-place machines.

## 2 Shift start checks
2.1 At the start of every shift the operator runs the nozzle check program. Any nozzle reported as worn or
blocked is replaced before production starts.
2.2 The operator confirms that the vacuum level reading of every head is within the green band on the machine
screen.
2.3 The shift start checks are recorded in the MES equipment log with the operator ID.

## 3 Feeder setup verification
3.1 Every feeder is scanned against the setup sheet before a work order starts. The MES blocks track-in if a
reel does not match the bill of materials.
3.2 A reel change during a lot must be verified by a second operator scan.

## 4 First-article inspection
4.1 The first board of each work order is placed and then inspected under the microscope against the assembly
drawing before the rest of the lot is released.
4.2 First-article inspection is also required after any machine calibration or nozzle change.

## 5 Placement offset alarms
5.1 The machine reports the placement offset of every fiducial-corrected component.
5.2 If the average X or Y offset over 50 placements exceeds 0.05 mm, the machine raises an offset warning.
5.3 If the warning repeats twice within one shift, the operator must stop the machine and call maintenance for a
head calibration (see SOP-MNT-008).

## 6 Moisture-sensitive devices
6.1 Moisture-sensitive components of level 3 have a floor life of 168 hours after the dry pack is opened.
6.2 Components that exceed their floor life must be baked at 125 degC for 24 hours before use.
