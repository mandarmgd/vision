# api.py
import cv2 as cv
import numpy as np
import json
from collections import Counter
from typing import List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Image Color Classifier API",
    description="Upload an image to classify it as 'rust', 'zinc', or 'normal' based on color heuristics."
)

# --- Define the Response Model (for OpenAPI docs and validation) ---
class ClassificationResponse(BaseModel):
    filename: str
    classification: str
    rustish_ratio: float
    zincish_ratio: float
    top_colors_rgb: List[List[int]]
    top_colors_share: List[float]


# --- Helper Functions (Copied from your main.py) ---

# Color space conversions
def bgr_to_rgb(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
def bgr_to_hsv(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
def bgr_to_lab(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2LAB)

# Dominant color extraction using KMeans
def dominant_colors_kmeans(bgr, k=3, max_iter=10):
    data = bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    _, labels, centers = cv.kmeans(data, k, None, criteria, 3, flags)
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
    counts = Counter(labels.flatten())
    total = float(len(labels))

    idx_sorted = [i for i, _ in counts.most_common()]
    palette = []
    for idx in idx_sorted:
        bgr_c = centers_u8[idx].tolist()
        rgb_c = bgr_to_rgb(np.array([[bgr_c]], dtype=np.uint8)).reshape(-1).tolist()
        share = counts[idx] / total
        palette.append({"share": float(share), "RGB": [int(x) for x in rgb_c]})
    return palette

# Heuristic calculation for rust/zinc
def rust_zinc_indicators(bgr, delta=6.0):
    lab = bgr_to_lab(bgr)
    _, a, b = cv.split(lab)
    a_med, b_med = np.median(a), np.median(b)
    a_thr = a_med + delta
    b_thr = b_med + delta

    rustish = (a.astype(np.float32) > a_thr).mean()
    zincish = (b.astype(np.float32) > b_thr).mean()
    return {"rustish_ratio": float(rustish), "zincish_ratio": float(zincish)}

# Classification logic
def classify_from_ratios(rustish_ratio, zincish_ratio, rust_thr=0.01, zinc_thr=0.02):
    if zincish_ratio > zinc_thr:
        return "zinc"
    elif rustish_ratio > rust_thr:
        return "rust"
    else:
        return "normal"

# --- API Endpoint ---

@app.post("/classify/", response_model=ClassificationResponse)
async def classify_image(
    file: UploadFile = File(..., description="The image file to classify."),
    k: int = Query(3, description="Number of dominant colors to extract."),
    rust_thr: float = Query(0.01, description="Threshold for 'rust' classification."),
    zinc_thr: float = Query(0.02, description="Threshold for 'zinc' classification."),
    lab_delta: float = Query(6.0, description="Sensitivity for heuristic indicators in Lab color space.")
):
    """
    Accepts an image file and returns a classification based on color analysis.
    """
    # 1. Read image bytes from upload
    contents = await file.read()
    
    # 2. Convert bytes to a NumPy array and then to an OpenCV image
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.")

    # 3. Perform color analysis and classification
    indicators = rust_zinc_indicators(bgr, delta=lab_delta)
    classification = classify_from_ratios(
        rustish_ratio=indicators["rustish_ratio"],
        zincish_ratio=indicators["zincish_ratio"],
        rust_thr=rust_thr,
        zinc_thr=zinc_thr
    )
    palette = dominant_colors_kmeans(bgr, k=max(1, k))

    # 4. Format the response
    response_data = {
        "filename": file.filename,
        "classification": classification,
        "rustish_ratio": round(indicators["rustish_ratio"], 4),
        "zincish_ratio": round(indicators["zincish_ratio"], 4),
        "top_colors_rgb": [p["RGB"] for p in palette],
        "top_colors_share": [round(p["share"], 4) for p in palette]
    }

    return response_data