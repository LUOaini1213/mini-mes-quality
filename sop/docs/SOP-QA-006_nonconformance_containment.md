# SOP-QA-006 Nonconformance, Containment and Root-Cause Analysis

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: C (synthetic) | Applies to: whole line

## 1 Purpose
This procedure defines when an abnormal defect rate triggers containment, who is notified and how the root
cause is investigated.

## 2 Triggers
2.1 The AOI confirmed-defect rate is monitored per hour against the baseline of the previous four weeks.
2.2 If the confirmed-defect rate is more than twice the baseline for two consecutive hours, a nonconformance
report (NCR) is opened.
2.3 A single confirmed defect of a safety-critical feature (for example a short circuit on the battery
protection circuit) opens an NCR immediately.

## 3 Containment
3.1 The shift leader must notify the quality engineer within 1 hour of an NCR being opened.
3.2 Suspect equipment is put on HOLD in the MES until the investigation releases it.
3.3 Suspect material batches are quarantined; the MES then holds every lot that consumed them (see WI-MES-005).

## 4 Root-cause analysis
4.1 The investigation uses the MES genealogy to compare defect rates by equipment, shift, operator and material
batch.
4.2 A factor is only accepted as a root cause when its effect remains after accounting for the other suspected
factors; a factor that is merely used at the same time as the real cause is not a root cause.
4.3 The 8D report must be completed within 10 working days.

## 5 Closure
5.1 An NCR is closed when the corrective action has run for two weeks without the trigger in section 2 repeating.
