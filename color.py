import cv2 as cv
import numpy as np
import argparse, os, json
from collections import Counter

def bgr_to_rgb(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
def bgr_to_hsv(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
def bgr_to_lab(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2LAB)

def img_stats(img, space_name):
    # img is uint8, shape HxWxC
    means = img.reshape(-1, img.shape[2]).mean(axis=0)
    stds  = img.reshape(-1, img.shape[2]).std(axis=0)
    return {
        "space": space_name,
        "mean": [float(x) for x in means],
        "std":  [float(x) for x in stds]
    }

def dominant_colors_kmeans(bgr, k=3, max_iter=10, seed=123):
    # reshape to N x 3
    data = bgr.reshape((-1, 3)).astype(np.float32)

    # kmeans
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    compactness, labels, centers = cv.kmeans(data, k, None, criteria, 3, flags)

    # centers are BGR float; convert to uint8
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
    counts = Counter(labels.flatten())
    total = float(len(labels))

    # sort by frequency desc
    idx_sorted = [i for i,_ in counts.most_common()]
    palette = []
    for idx in idx_sorted:
        bgr_c = centers_u8[idx].tolist()
        rgb_c = bgr_to_rgb(np.array([[bgr_c]], dtype=np.uint8)).reshape(-1).tolist()
        hsv_c = bgr_to_hsv(np.array([[bgr_c]], dtype=np.uint8)).reshape(-1).tolist()
        lab_c = bgr_to_lab(np.array([[bgr_c]], dtype=np.uint8)).reshape(-1).tolist()
        share = counts[idx] / total
        palette.append({
            "share": float(share),
            "BGR": [int(x) for x in bgr_c],
            "RGB": [int(x) for x in rgb_c],
            "HSV": [int(x) for x in hsv_c],
            "Lab": [int(x) for x in lab_c],
        })
    return palette

def make_palette_image(palette, width=600, height=80, pad=2):
    # palette: list of dicts with 'share' and 'RGB'
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for p in palette:
        w = max(1, int(p["share"] * width))
        color = tuple(p["RGB"])  # RGB
        # convert to BGR for OpenCV drawing
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv.rectangle(img, (x, 0), (min(width-1, x+w-1), height-1), bgr, -1)
        x += w
    # thin separators
    for i in range(1, len(palette)):
        x_sep = int(sum([pp["share"] for pp in palette[:i]]) * width)
        cv.line(img, (x_sep, 0), (x_sep, height-1), (30,30,30), 1)
    return img

def rust_zinc_indicators(bgr):
    """Heuristic only, NO detection claims. Gives ratios based on Lab chroma tendencies:
       - 'rustish_ratio': fraction of pixels with a* significantly above median (reddish/brownish)
       - 'zincish_ratio': fraction of pixels with b* significantly above median (yellowish)
    """
    lab = bgr_to_lab(bgr)
    L, a, b = cv.split(lab)
    a_med, b_med = np.median(a), np.median(b)
    a_thr = a_med + 6  # tweak if needed
    b_thr = b_med + 6

    rustish = (a.astype(np.float32) > a_thr).mean()
    zincish = (b.astype(np.float32) > b_thr).mean()
    return {"rustish_ratio": float(rustish), "zincish_ratio": float(zincish),
            "a_median": float(a_med), "b_median": float(b_med),
            "a_thresh": float(a_thr), "b_thresh": float(b_thr)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to image")
    ap.add_argument("--k", type=int, default=3, help="number of dominant colors")
    ap.add_argument("--resize_max", type=int, default=1200, help="resize longer side to this (0=off)")
    ap.add_argument("--outdir", default="color_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bgr = cv.imread(args.img, cv.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {args.img}")

    # Optional resize to speed up
    h, w = bgr.shape[:2]
    if args.resize_max > 0:
        s = max(h, w)
        if s > args.resize_max:
            scale = args.resize_max / float(s)
            bgr = cv.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)

    # Color-space stats
    rgb = bgr_to_rgb(bgr)
    hsv = bgr_to_hsv(bgr)
    lab = bgr_to_lab(bgr)

    stats = [
        img_stats(rgb, "RGB"),  # channels: R,G,B (0-255)
        img_stats(hsv, "HSV"),  # channels: H(0-179), S(0-255), V(0-255) in OpenCV
        img_stats(lab, "Lab"),  # channels: L(0-255), a(0-255), b(0-255) in OpenCV's scaled Lab
    ]

    # Dominant colors (k-means)
    palette = dominant_colors_kmeans(bgr, k=max(1, args.k))

    # Heuristic indicators (optional)
    indicators = rust_zinc_indicators(bgr)

    # Save palette image
    pal_img = make_palette_image(palette)
    base = os.path.splitext(os.path.basename(args.img))[0]
    pal_path = os.path.join(args.outdir, f"{base}_palette.png")
    cv.imwrite(pal_path, pal_img)

    # Build and save JSON
    report = {
        "input": os.path.basename(args.img),
        "size_hw": [int(bgr.shape[0]), int(bgr.shape[1])],
        "color_stats": stats,
        "dominant_colors": palette,  # ordered by share desc
        "heuristics": indicators,
        "palette_image": pal_path
    }
    rep_path = os.path.join(args.outdir, f"{base}_color_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print a short summary to console
    print(json.dumps({
        "input": report["input"],
        "top_colors_rgb": [p["RGB"] for p in report["dominant_colors"]],
        "top_colors_share": [round(p["share"], 4) for p in report["dominant_colors"]],
        "rustish_ratio": round(report["heuristics"]["rustish_ratio"], 4),
        "zincish_ratio": round(report["heuristics"]["zincish_ratio"], 4),
        "report_path": rep_path,
        "palette_image": pal_path
    }, indent=2))

if __name__ == "__main__":
    main()
