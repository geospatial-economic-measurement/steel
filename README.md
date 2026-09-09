# Replication package

**A Geospatial Approach to Measuring Economic Activity**

Yang, Ai & Arkolakis

## Instructions

```bash
cd code
pip install -r requirements.txt
python 00_run_all.py
```

Figures go to `output/figures`, tables to `output/tables`. All paths are relative to the package.

## Structure

```
code/     scripts; _paths.py defines all locations
data/     inputs (processed/confidential/ is restricted — see below)
output/   figures/ and tables/ — everything in the paper and SI
output_rl/ diagnostics; not part of the submission
```

`output/figures` holds one file per figure, named by its number:
`Figure_01a.png` … `Figure_09b.png` (paper), `Figure_S01.pdf` …
`Figure_S33.png` (Supplementary).

## Data

Two groups are excluded from the public archive:

|                                                               | What                            | How to get it                                                                            |
| ------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------- |
| `data/processed/confidential/`                              | CISA production labels, 5 files | Licensed commercial data — access via[mysteel.com](https://www.mysteel.com)              |
| `data/SK`, `NK`, `IR`, `first/second_output_file.csv` | satellite panels                | Rebuild with scripts`28`, `29`, `35`, `39` (needs a Google Earth Engine account) |

`00_run_all.py` skips whatever depends on confidential data.

## Requirements

Python 3.13 · pandas 2.3.3 · numpy 2.4.2 · scikit-learn 1.8.0 · xgboost 3.2.0 ·
tensorflow 2.21.0 · shap 0.51.0 · matplotlib 3.10.8 · scipy · umap-learn ·
cartopy · openpyxl. `earthengine-api` only for the download scripts.
Exact pins in `code/requirements.txt`.

## Which script makes which exhibit

| Exhibit            | File(s) in`output/figures`                              | Script      |
| ------------------ | --------------------------------------------------------- | ----------- |
| **Figure 1** | `Figure_01a.png`, `Figure_01b.jpg`                    | `08` |
| **Figure 2** | `Figure_02.pdf`                                         | `04`      |
| **Figure 3** | `Figure_03a.pdf`, `Figure_03b.pdf`                    | `11 / 09` |
| **Figure 4** | `Figure_04.pdf`                                         | `12`      |
| **Figure 5** | `Figure_05a.pdf`, `Figure_05b.pdf`                    | `13`      |
| **Figure 6** | `Figure_06a.pdf`, `Figure_06b.pdf`                    | `24`      |
| **Figure 7** | `Figure_07a.pdf`, `Figure_07b.pdf`                    | `22 / 23` |
| **Figure 8** | `Figure_08a.png`, `Figure_08b.pdf`                    | `31 / 34` |
| **Figure 9** | `Figure_09a.png`, `Figure_09b.png`                    | `36 / 38` |
| **Table 1**  | `output/tables/Table_01_sk_validation_probweighted.csv` | `34`      |

| Supplementary         | File(s)                                                                            | Script                     |
| --------------------- | ---------------------------------------------------------------------------------- | -------------------------- |
| Supplementary Fig. 1  | `Figure_S01.pdf`                                                                 | `03`                     |
| Supplementary Fig. 2  | `Figure_S02a.pdf`, `Figure_S02b.pdf`                                           | `06`                     |
| Supplementary Fig. 3  | `Figure_S03a.pdf`, `Figure_S03b.pdf`                                           | `02`                     |
| Supplementary Fig. 4  | `Figure_S04.pdf`                                                                 | `27`                     |
| Supplementary Fig. 5  | `Figure_S05.pdf`                                                                 | `11`                     |
| Supplementary Fig. 6  | `Figure_S06a.png`, `Figure_S06b.png`, `Figure_S06c.png`, `Figure_S06d.png` | `nb 52`                  |
| Supplementary Fig. 7  | `Figure_S07.jpg`                                                                 | `nb 53 (R)`              |
| Supplementary Fig. 8  | `Figure_S08a.png`, `Figure_S08b.png`, `Figure_S08c.png`                      | `– (satellite imagery)` |
| Supplementary Fig. 9  | `Figure_S09.pdf`                                                                 | `nb 54`                  |
| Supplementary Fig. 10 | `Figure_S10.pdf`                                                                 | `07`                     |
| Supplementary Fig. 11 | `Figure_S11.pdf`                                                                 | `nb 56`                  |
| Supplementary Fig. 12 | `Figure_S12.pdf`                                                                 | `nb 55`                  |
| Supplementary Fig. 13 | `Figure_S13.pdf`                                                                 | `14`                     |
| Supplementary Fig. 14 | `Figure_S14.pdf`                                                                 | `nb 57`                  |
| Supplementary Fig. 15 | `Figure_S15.pdf`                                                                 | `nb 58`                  |
| Supplementary Fig. 16 | `Figure_S16.pdf`                                                                 | `17`                     |
| Supplementary Fig. 17 | `Figure_S17.pdf`                                                                 | `18`                     |
| Supplementary Fig. 18 | `Figure_S18.pdf`                                                                 | `nb 58`                  |
| Supplementary Fig. 19 | `Figure_S19.pdf`                                                                 | `18`                     |
| Supplementary Fig. 20 | `Figure_S20.pdf`                                                                 | `nb 58`                  |
| Supplementary Fig. 21 | `Figure_S21.pdf`                                                                 | `19`                     |
| Supplementary Fig. 22 | `Figure_S22a.pdf`, `Figure_S22b.pdf`, `Figure_S22c.pdf`, `Figure_S22d.pdf` | `16`                     |
| Supplementary Fig. 23 | `Figure_S23.pdf`                                                                 | `–`                     |
| Supplementary Fig. 24 | `Figure_S24.pdf`                                                                 | `–`                     |
| Supplementary Fig. 26 | `Figure_S26.pdf`                                                                 | `–`                     |
| Supplementary Fig. 27 | `Figure_S27a.png`, `Figure_S27b.png`, `Figure_S27c.jpg`, `Figure_S27d.jpg` | `– (maps)`              |
| Supplementary Fig. 28 | `Figure_S28.pdf`                                                                 | `–`                     |
| Supplementary Fig. 29 | `Figure_S29.pdf`                                                                 | `–`                     |
| Supplementary Fig. 31 | `Figure_S31.pdf`                                                                 | `–`                     |
| Supplementary Fig. 32 | `Figure_S32a.png`, `Figure_S32b.png`                                           | `40`                     |
| Supplementary Fig. 33 | `Figure_S33.png`                                                                 | `40`                     |

`–` = shipped output with no generating script in this package (manual imagery or policy simulations). `nb NN` = notebook or R script in `code/notebooks/`, run interactively; see the README there.

## Pipeline order

`00_run_all.py` runs these in order. ‡ needs the restricted CISA data.

1. **Location classifier** — `01`‡ `02`‡ `03` `04`‡ `05` `06` `07` `08` `09` `10`‡ `11`
2. **Grid output model** — `12`‡ `13`‡ `14`‡ `15`‡ `16`‡
3. **Plant output model** — `17`‡ `18`‡ `19`‡ `20`‡
4. **Validation and events** — `21`‡ `22`‡ `23`‡ `24`‡ `25`‡ `26`‡ `27`‡
5. **South Korea** — `28` `29` `30` `31`‡ `32` `34` `33`
6. **Iran** — `35` `36` `37` `38`
7. **North Korea** — `39` `40`
8. **Robustness** — `41`‡ `42`‡ `43`‡ `44`‡ `45`‡ `46`‡ `47`‡ `48`‡

Two ordering constraints: `10` before `11`, and `01`/`02` before `03`/`05`/`06`/`09`.
