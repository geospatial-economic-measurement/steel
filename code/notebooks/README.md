# Notebook producers for Supplementary figures

Original notebooks (outputs stripped) and one R script that generate nine Supplementary figures.

| File                                     | Produces             | Notes                                                                                                                                                                            |
| ---------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `52_chap_s5p_density_comparison.ipynb` | Supp Fig 6           | CHAP vs Sentinel-5P NO2 KDE density panels, one per year; reads the CHAP and converted S5P GeoTIFFs                                                                              |
| `53_steel_output_sources.R`            | Supp Fig 7           | NBS vs CISA vs large/medium-plant monthly production; reads`Steel_NBS.xlsx`, `Steel_CISA.xlsx`, `CISA_largemedium_plant.xlsx`                                              |
| `54_plant_area_distribution.ipynb`     | Supp Fig 9           | Plant-area histogram, all vs operating GEM plants                                                                                                                                |
| `55_grid_spatial_heterogeneity.ipynb`  | Supp Fig 12          | Nine satellite signals over time for grid cells with vs without steel plants                                                                                                     |
| `56_china_output_map.ipynb`            | Supp Fig 11          | Folium map on Esri World Imagery tiles, plants coloured by production ("Steel Production (tppa)" colormap)                                                                       |
| `57_geoshapley_grid.ipynb`             | Supp Fig 14          | Grid-level model pipeline (fit, actual-vs-predicted, SHAP, PDP). Run on`data/processed/sentinel_notebook_inputs/Factory_Emission_And_LST_Grid.csv` for the Sentinel-5P version |
| `58_geoshapley_plant.ipynb`            | Supp Figs 15, 18, 20 | Plant-level pipeline (actual-vs-predicted, SHAP, bootstrap feature importance). Run on the`Factory_Emission_And_LST1/_lag` + `mode.csv` Sentinel inputs                      |
