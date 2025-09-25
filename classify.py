import json, argparse

def classify_from_ratios(rustish_ratio, zincish_ratio, rust_thr=0.01, zinc_thr=0.02):
    if zincish_ratio > zinc_thr:
        return "zinc"
    elif rustish_ratio > rust_thr:
        return "rust"
    else:
        return "normal"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="path to *_color_report.json from color.py")
    ap.add_argument("--rust_thr", type=float, default=0.01)
    ap.add_argument("--zinc_thr", type=float, default=0.02)
    args = ap.parse_args()

    with open(args.report, "r") as f:
        rep = json.load(f)

    rustish_ratio = rep["heuristics"]["rustish_ratio"]
    zincish_ratio = rep["heuristics"]["zincish_ratio"]

    classification = classify_from_ratios(rustish_ratio, zincish_ratio,
                                          rust_thr=args.rust_thr,
                                          zinc_thr=args.zinc_thr)

    print({
        "file": rep["input"],
        "rustish_ratio": rustish_ratio,
        "zincish_ratio": zincish_ratio,
        "classification": classification
    })
