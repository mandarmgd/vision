import gradio as gr
import cv2 as cv
import numpy as np
from api import rust_zinc_indicators, classify_from_ratios, dominant_colors_kmeans

def classify_image_gradio(image, k=3, rust_thr=0.01, zinc_thr=0.02, lab_delta=6.0):
    # OpenCV expects BGR
    bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Run your existing functions
    indicators = rust_zinc_indicators(bgr, delta=lab_delta)
    classification = classify_from_ratios(
        rustish_ratio=indicators["rustish_ratio"],
        zincish_ratio=indicators["zincish_ratio"],
        rust_thr=rust_thr,
        zinc_thr=zinc_thr
    )
    palette = dominant_colors_kmeans(bgr, k=max(1, k))

    return {
        "classification": classification,
        "rustish_ratio": round(indicators["rustish_ratio"], 4),
        "zincish_ratio": round(indicators["zincish_ratio"], 4),
        "top_colors_rgb": [p["RGB"] for p in palette],
        "top_colors_share": [round(p["share"], 4) for p in palette]
    }

# Gradio UI
demo = gr.Interface(
    fn=classify_image_gradio,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Slider(1, 6, step=1, value=3, label="Dominant Colors (k)"),
        gr.Slider(0.0, 0.1, step=0.001, value=0.01, label="Rust Threshold"),
        gr.Slider(0.0, 0.1, step=0.001, value=0.02, label="Zinc Threshold"),
        gr.Slider(1.0, 20.0, step=0.5, value=6.0, label="Lab Delta")
    ],
    outputs="json",
    title="Rust vs Zinc Classifier",
    description="Upload an image to classify it as 'rust', 'zinc', or 'normal' based on color heuristics."
)

if __name__ == "__main__":
    demo.launch(share=True)
