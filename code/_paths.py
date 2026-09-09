import os

CODE = os.path.dirname(os.path.abspath(__file__))
REP_ROOT = os.path.dirname(CODE)

DATA = os.path.join(REP_ROOT, "data")
DATA_PROCESSED = os.path.join(DATA, "processed")
DATA_CONFIDENTIAL = os.path.join(DATA_PROCESSED, "confidential")
DATA_RAW = os.path.join(DATA, "raw")
DATA_SK = os.path.join(DATA, "SK")
DATA_NK = os.path.join(DATA, "NK")
DATA_IR = os.path.join(DATA, "IR")
CODEBOOKS = os.path.join(DATA, "codebooks")
MODEL_ARTIFACTS = os.path.join(CODE, "model_artifacts")

OUTPUT = os.path.join(REP_ROOT, "output")
OUT_FIG = os.path.join(OUTPUT, "figures")
OUT_TAB = os.path.join(OUTPUT, "tables")
OUT_TAB_INTERMEDIATE = os.path.join(OUT_TAB, "intermediate")
OUT_RL = os.path.join(REP_ROOT, "output_rl")
SCRATCH = os.path.join(REP_ROOT, "scratch")

for _d in (OUT_FIG, OUT_TAB, OUT_TAB_INTERMEDIATE, OUT_RL, SCRATCH):
    os.makedirs(_d, exist_ok=True)

PAPER_FIG = os.environ.get("NC_STEEL_PAPER_FIG") or None

RESTRICTED = {
    "GridProd_1922_monthly.csv", "gridinfo_xy.csv", "Location_Plants_Full.xlsx",
    "re_data.csv", "plantmode_ml.csv",
}


def need(*parts):
    """Path to a required input; raises with a readable message if absent."""
    p = os.path.join(*parts)
    if os.path.exists(p):
        return p
    msg = "Required input not found:\n  %s" % p
    if os.path.basename(p) in RESTRICTED:
        msg += ("\n  Restricted CISA data, not in the public archive."
                "\n  See the 'Data you need' section of the README.")
    raise FileNotFoundError(msg)


def first_existing(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("None of these exist:\n  " +
                            "\n  ".join(str(c) for c in candidates))


def save_fig(fig, name, exts=("pdf",), dpi=300, also_paper=True):
    """Save a figure into output/figures."""
    for ext in exts:
        p = os.path.join(OUT_FIG, "%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print("  saved %s" % p)
        if also_paper and PAPER_FIG:
            try:
                fig.savefig(os.path.join(PAPER_FIG, "%s.%s" % (name, ext)),
                            bbox_inches="tight", dpi=dpi)
            except OSError:
                pass


def save_table(df, name, intermediate=False, **kw):
    """Save a table into output/tables."""
    d = OUT_TAB_INTERMEDIATE if intermediate else OUT_TAB
    p = os.path.join(d, name if name.endswith(".csv") else name + ".csv")
    kw.setdefault("index", False)
    kw.setdefault("encoding", "utf-8-sig")
    df.to_csv(p, **kw)
    print("  saved %s" % p)
    return p


def save_diagnostic(fig, name, exts=("pdf", "png"), dpi=300):
    """Save a figure that is not in the paper into output_rl."""
    for ext in exts:
        p = os.path.join(OUT_RL, "%s.%s" % (name, ext))
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print("  saved %s" % p)


if __name__ == "__main__":
    print("REP_ROOT  %s" % REP_ROOT)
    for k in ("DATA", "DATA_PROCESSED", "DATA_CONFIDENTIAL", "DATA_RAW", "DATA_SK",
              "DATA_NK", "DATA_IR", "CODEBOOKS", "MODEL_ARTIFACTS", "OUT_FIG",
              "OUT_TAB", "OUT_RL", "SCRATCH"):
        v = globals()[k]
        print("%-18s %-4s %s" % (k, "OK" if os.path.exists(v) else "MISS", v))
