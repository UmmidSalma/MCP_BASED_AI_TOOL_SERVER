"""UltraCamNet (cancer detection) tool.

DEMO VERSION: returns a mock prediction after checking the file exists.
TODO: load your real model and run inference + Grad-CAM, e.g.
      model = torch.load("models/ultracamnet.pt"); ...
"""
import os


def predict_image(image_path: str) -> dict:
    if not os.path.isfile(image_path):
        return {"error": f"File not found: {image_path}"}

    # --- replace this block with real inference ---
    return {
        "image": os.path.basename(image_path),
        "prediction": "malignant (DEMO)",
        "confidence": 0.91,
        "gradcam_path": None,  # TODO: save heatmap and return its path
        "note": "Placeholder output. Plug in the real UltraCamNet model.",
    }
