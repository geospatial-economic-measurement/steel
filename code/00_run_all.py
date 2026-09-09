"""Run the pipeline in order.

    python 00_run_all.py [--list] [--step N] [--only 13c] [--with-gee]
                         [--paper-figs] [--continue-on-error]

Scripts whose inputs are missing (restricted data, satellite panels, GEE) are
skipped with a reason.
"""
import argparse
import os
import subprocess
import sys
import time

from _paths import CODE, DATA_CONFIDENTIAL, OUT_FIG, OUT_TAB, REP_ROOT

# step -> (title, [scripts])
PIPELINE = [
    ("1. Location classifier", [
        "01_location_classifier_train_shap.py",
        "02_location_classifier_eval_full_population.py",
        "03_location_classifier_negative_control_umap.py",
        "04_location_classifier_coverage_ratio_plot.py",
        "05_plant_type_clustering_sensitivity.py",
        "06_plant_type_clustering_extensions.py",
        "07_location_classifier_correlation_matrix.py",
        "08_fig1a_prediction_map.py",
        "09_location_classifier_plot_shap.py",
        # 04a must precede 04: it writes table_month_retrain_robustness.csv,
        # which 04 reads to build the season-averaged confusion matrix.
        "10_location_classifier_month_retrain_robustness.py",
        "11_location_classifier_seasonal_cm_figure.py",
    ]),
    ("2. Grid-level output model", [
        "12_grid_output_model_fit_figure.py",
        "13_grid_output_shap_figures.py",
        "14_grid_output_pdp_figure.py",
        "15_grid_output_model_comparison_table.py",
        "16_grid_plant_output_model_boxplots.py",
    ]),
    ("3. Plant-level output model", [
        "17_plant_output_model_fit_figure.py",
        "18_plant_output_shap_figures.py",
        "19_plant_output_pdp_figure.py",
        "20_plant_output_model_comparison_table.py",
    ]),
    ("4. Validation and event analysis", [
        "21_holdout_validation_temporal_geo_figures.py",
        "22_temporal_holdout_paper.py",
        "23_geo_holdout_south_full_symlog_paper.py",
        "24_event_analysis_springfestival_covid_figures.py",
        "25_loyocv_r2_table.py",
        "26_cs_temporal_r2_decomposition_table.py",
        "27_growth_rate_predictions.py",
    ]),
    ("5. South Korea", [
        "28_sk_china_sentinel_download_gee.py",
        "29_sk_china_sentinel_2018q4_patch_gee.py",
        "30_sk_sentinel_model_train_apply.py",
        "31_sk_location_tech_stratified.py",
        "32_sk_output_validation_figures.py",
        "33_sk_growth_bars_paper.py",
        "34_sk_growth_bars_plant.py",
    ]),
    ("6. Iran", [
        "35_ua_ir_download_gee.py",
        "36_iran_location_classifier.py",
        "37_iran_output_monthly_figure.py",
        "38_ir_annotated_monthly_yoy.py",
    ]),
    ("7. North Korea", [
        "39_nk_download_prepare_gee.py",
        "40_nk_figures.py",
    ]),
    ("8. Robustness and supporting tables", [
        "41_city_loo.py",
        "42_figR22_year_by_year_bars.py",
        "43_spatial_block_cv_crosssectional.py",
        "44_validation_summary_table.py",
        "45_within_year_crosssectional.py",
        "46_within_year_split.py",
        "47_grid_plant_agg_5fold_confirmed.py",
        "48_year_by_year_ablation.py",
    ]),
]

GEE = {"28_sk_china_sentinel_download_gee.py",
       "29_sk_china_sentinel_2018q4_patch_gee.py",
       "35_ua_ir_download_gee.py",
       "39_nk_download_prepare_gee.py"}

# scripts that cannot run without the restricted CISA inputs
NEEDS_CONFIDENTIAL = {
    "01_location_classifier_train_shap.py", "02_location_classifier_eval_full_population.py",
    "04_location_classifier_coverage_ratio_plot.py",
    "10_location_classifier_month_retrain_robustness.py",
    "12_grid_output_model_fit_figure.py", "13_grid_output_shap_figures.py",
    "14_grid_output_pdp_figure.py", "15_grid_output_model_comparison_table.py",
    "16_grid_plant_output_model_boxplots.py", "17_plant_output_model_fit_figure.py",
    "18_plant_output_shap_figures.py", "19_plant_output_pdp_figure.py",
    "20_plant_output_model_comparison_table.py",
    "21_holdout_validation_temporal_geo_figures.py", "22_temporal_holdout_paper.py",
    "23_geo_holdout_south_full_symlog_paper.py",
    "24_event_analysis_springfestival_covid_figures.py", "25_loyocv_r2_table.py",
    "26_cs_temporal_r2_decomposition_table.py", "27_growth_rate_predictions.py",
    "31_sk_location_tech_stratified.py", "41_city_loo.py",
    "42_figR22_year_by_year_bars.py", "43_spatial_block_cv_crosssectional.py",
    "44_validation_summary_table.py", "45_within_year_crosssectional.py",
    "46_within_year_split.py", "47_grid_plant_agg_5fold_confirmed.py",
    "48_year_by_year_ablation.py",
}

HAVE_CONFIDENTIAL = os.path.exists(
    os.path.join(DATA_CONFIDENTIAL, "GridProd_1922_monthly.csv"))

# Scripts that read the large country panels. Those panels are excluded from
# the public archive because they are rebuilt by the GEE download scripts, so
# skip cleanly rather than letting the script die on a missing file.
from _paths import DATA, DATA_IR, DATA_NK, DATA_SK

NEEDS_PANEL = {
    "30_sk_sentinel_model_train_apply.py": DATA_SK,
    "31_sk_location_tech_stratified.py": DATA_SK,
    "32_sk_output_validation_figures.py": DATA_SK,
    "33_sk_growth_bars_paper.py": DATA_SK,
    "34_sk_growth_bars_plant.py": DATA_SK,
    "36_iran_location_classifier.py": DATA_IR,
    "37_iran_output_monthly_figure.py": DATA_IR,
    "38_ir_annotated_monthly_yoy.py": DATA_IR,
    "40_nk_figures.py": DATA_NK,
}

# the two national grid-month panels the location classifier trains on
NEEDS_RAW_PANELS = {
    "01_location_classifier_train_shap.py",
    "02_location_classifier_eval_full_population.py",
    "04_location_classifier_coverage_ratio_plot.py",
    "10_location_classifier_month_retrain_robustness.py",
}
HAVE_RAW_PANELS = os.path.exists(
    os.path.join(os.path.dirname(DATA_CONFIDENTIAL), "first_output_file.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--step", type=int)
    ap.add_argument("--only")
    ap.add_argument("--with-gee", action="store_true")
    ap.add_argument("--paper-figs", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    a = ap.parse_args()

    if a.paper_figs:
        os.environ["NC_STEEL_PAPER_FIG"] = os.path.join(
            os.path.dirname(REP_ROOT), "Final", "Figure")

    print("replication root : %s" % REP_ROOT)
    print("figures -> %s" % OUT_FIG)
    print("tables  -> %s" % OUT_TAB)
    print("restricted CISA data present: %s" % ("yes" if HAVE_CONFIDENTIAL else "no"))
    print()

    ran, skipped, failed = [], [], []
    for i, (title, scripts) in enumerate(PIPELINE, 1):
        if a.step and i != a.step:
            continue
        print("=" * 66)
        print(title)
        print("=" * 66)
        for s in scripts:
            if a.only and a.only not in s:
                continue
            if s in GEE and not a.with_gee:
                skipped.append((s, "needs Google Earth Engine (--with-gee)"))
                print("  SKIP %-52s GEE" % s)
                continue
            if s in NEEDS_CONFIDENTIAL and not HAVE_CONFIDENTIAL:
                skipped.append((s, "needs restricted CISA data"))
                print("  SKIP %-52s restricted data" % s)
                continue
            if s in NEEDS_RAW_PANELS and not HAVE_RAW_PANELS:
                skipped.append((s, "needs first/second_output_file.csv (rebuild with GEE)"))
                print("  SKIP %-52s raw panels absent" % s)
                continue
            d = NEEDS_PANEL.get(s)
            if d and not os.path.isdir(d):
                skipped.append((s, "needs %s (rebuild with the GEE scripts)"
                                % os.path.basename(d)))
                print("  SKIP %-52s %s absent" % (s, os.path.basename(d)))
                continue
            if not os.path.exists(os.path.join(CODE, s)):
                skipped.append((s, "script not found"))
                print("  SKIP %-52s missing" % s)
                continue

            if a.list:
                print("  RUN  %s" % s)
                continue

            t0 = time.time()
            print("  RUN  %-52s " % s, end="", flush=True)
            # utf-8 with replacement: several scripts print Chinese plant names,
            # which the default Windows codepage (gbk) cannot decode.
            r = subprocess.run([sys.executable, s], cwd=CODE,
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            dt = time.time() - t0
            if r.returncode == 0:
                ran.append((s, dt))
                print("ok   %5.1fs" % dt)
            else:
                failed.append((s, r.returncode, (r.stderr or "").strip().split("\n")[-1]))
                print("FAIL %5.1fs" % dt)
                print("       %s" % failed[-1][2][:100])
                if not a.continue_on_error:
                    print("\nstopping (use --continue-on-error to keep going)")
                    break
        else:
            continue
        break

    print("\n" + "=" * 66)
    print("ran %d | skipped %d | failed %d" % (len(ran), len(skipped), len(failed)))
    if skipped:
        print("\nskipped:")
        for s, why in skipped:
            print("   %-52s %s" % (s, why))
    if failed:
        print("\nfailed:")
        for s, rc, msg in failed:
            print("   %-52s rc=%s %s" % (s, rc, msg[:60]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
