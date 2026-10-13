# SOP-AOI-004 Automated Optical Inspection and Disposition

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: E (synthetic) | Applies to: AOI-01, AOI-02 | Route step: 40 AOI

## 1 Purpose
This procedure covers inspection of soldered boards by the AOI stations, the review of AOI calls and the
disposition of failed units.

## 2 Model and threshold control
2.1 The anomaly-detection model version and its decision threshold are controlled items. Only the quality
engineer may change them, through a change request.
2.2 The model version used for every inspection is written to the inspection record in the MES.
2.3 After a model or threshold change, the golden-board set must be re-run and all known defective golden boards
must be called FAIL.

## 3 Daily golden-board check
3.1 At the start of every shift the operator runs the golden-board set: 5 good boards and 5 boards with known
defects.
3.2 If any golden board is called wrongly, the AOI station is put on HOLD and the quality engineer is informed.

## 4 Review of FAIL calls
4.1 Every unit called FAIL is reviewed by a trained inspector at the review station within 30 minutes.
4.2 The inspector classifies the call as a true defect or a false call and records the result in the MES.
4.3 An operator may not override a FAIL call to PASS. Only an inspector with review authorisation may do this,
and the reason must be recorded.

## 5 Disposition of defective units
5.1 Confirmed defective units are placed in the red MRB rack and their status is set to HOLD in the MES.
5.2 A defective unit may be repaired at most twice. After a second failed repair it is scrapped.
5.3 Repaired units must be re-inspected at AOI before they continue to assembly.

## 6 Image retention
6.1 Inspection images are kept for 90 days on the AOI server and then deleted automatically.
