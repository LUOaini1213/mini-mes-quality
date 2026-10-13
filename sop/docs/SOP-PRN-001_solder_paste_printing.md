# SOP-PRN-001 Solder Paste Printing

> SYNTHETIC DOCUMENT - written for the mini-mes-quality demo. Fictional line, fictional values. Not a real
> procedure and not derived from any company's documents. Do not use for real production.

Revision: C (synthetic) | Applies to: printers PRN-01, PRN-02 | Route step: 10 PRINT

## 1 Purpose
This procedure defines how solder paste is stored, prepared, loaded and printed on bare PCBs, and what must be
recorded in the MES at the PRINT step.

## 2 Paste storage and preparation
2.1 Unopened solder paste jars are stored in the paste refrigerator at 2 to 10 degC.
2.2 Before opening, a jar must warm up at room temperature for at least 4 hours. Do not heat paste to speed up
warm-up.
2.3 Stir the paste for 1 minute with the automatic mixer before first use.
2.4 An opened jar must be used within 24 hours of opening. Write the opening time on the jar label.
2.5 Paste that has been on the stencil for more than 8 hours must be scrapped and must not be returned to the jar.

## 3 Loading and batch scan
3.1 Scan the paste batch ID into the MES when the jar is loaded. The MES links the batch to every lot printed until
the next batch scan.
3.2 Scan the bare-board (PCB_BLANK) batch ID when a new board magazine is loaded.
3.3 A lot cannot be tracked in at PRINT if no paste batch has been scanned on that printer during the current
shift.

## 4 Printing parameters
4.1 Squeegee pressure is set between 5 and 7 kg for the 300 mm squeegee.
4.2 Print speed is set between 30 and 60 mm/s.
4.3 The stencil underside is cleaned automatically every 10 prints (wet-vacuum-dry cycle).
4.4 Solder paste inspection (SPI) checks every board. Paste volume must be within 70 to 150 percent of nominal.

## 5 Reaction to SPI failures
5.1 Three consecutive SPI failures stop the printer automatically.
5.2 The operator cleans the stencil, prints one test board and asks the shift technician to release the printer.
5.3 Boards that fail SPI are washed and may be reprinted once. A board may not be reprinted more than once.
