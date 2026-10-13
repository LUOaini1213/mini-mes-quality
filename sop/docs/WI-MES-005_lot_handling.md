# WI-MES-005 Lot Handling in the MES (Split, Merge, Hold)

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: A (synthetic) | Applies to: all route steps

## 1 Purpose
This work instruction explains how lots are created, split, merged and held in the MES so that every unit keeps
a complete genealogy.

## 2 Lot release
2.1 Lots are released from a work order with at most 96 units per lot.
2.2 Each unit receives a serial number at release; the serial is laser-marked on the board edge.

## 3 Lot split
3.1 A lot may be split only after the PLACE step has been tracked out.
3.2 Each child lot keeps the work order of the parent. The MES records the split as a genealogy event with the
quantity moved to each child.
3.3 Splitting is used to run the children on different reflow ovens or AOI stations when one is busy.

## 4 Lot merge
4.1 Lots may be merged only if they belong to the same work order and have completed the AOI step.
4.2 The merged lot records each parent lot and the quantity it contributed.
4.3 A lot on HOLD cannot be merged.

## 5 Hold and release
5.1 Any operator can put a lot on HOLD and must enter a reason code.
5.2 Only a quality engineer can release a lot from HOLD.
5.3 When a material batch is quarantined, the MES puts every lot that consumed the batch on HOLD, including
child and merged lots found through the genealogy.

## 6 Quantity reconciliation
6.1 At every track-out the good quantity plus the scrap quantity must equal the track-in quantity.
6.2 A difference of even one unit blocks track-out until it is explained.
