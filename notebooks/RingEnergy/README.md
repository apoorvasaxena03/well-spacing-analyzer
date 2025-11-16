# Floating Section Diagnostics

**Box:** 5280 ft × 5280 ft  
**Inside-length threshold:** 660 ft  
**Exclude self:** True

## What’s included
- Per-well maps (`map_<uwi>_<orientation>.png`) showing:
  - all laterals,
  - the active floating box (**black**) centered on the reference well,
  - the alternate box style (**gray**) for context,
  - labels for neighbors with inside length ≥ threshold (`<uwi>`, `<inside_len_ft> ft`).
- Neighbor tables (`neighbors_<uwi>_<orientation>.csv`) listing counted wells and inside lengths.
- Basin-wide polar azimuth histogram (`azimuth_polar.png`) for **stratification**.
- Mean WPS by azimuth bins chart (`wps_by_azimuth_bins.png`) for **normalization** (cardinal vs i-frame).

## Notes
- *i-frame* aligns the box with the reference well’s azimuth (0°=N, 90°=E).
- *cardinal* keeps the box north-up, east-right.
- A neighbor is counted only if its lateral length **inside** the box is ≥ the threshold.
