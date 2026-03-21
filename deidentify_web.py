"""
Endoscopy Image De-identification - Streamlit Web App
EasyOCR (CRAFT) + OpenCV inpainting
"""

import cv2
import numpy as np
import easyocr
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.1
INPAINT_RADIUS       = 10
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return easyocr.Reader(["en"], gpu=False)

def build_mask(image, results):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (bbox, text, conf) in results:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def deidentify(img):
    reader  = load_model()
    results = reader.readtext(img)
    mask    = build_mask(img, results)
    cleaned = cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    texts   = [text for (_, text, conf) in results if conf >= CONFIDENCE_THRESHOLD]
    return cleaned, texts

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Endoscopy Image De-identification")
st.markdown("Upload PNG images to automatically detect and remove burned-in patient data.")

uploaded_files = st.file_uploader(
    "Select images", type=["png", "jpg"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(uploaded_file.name)

        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with st.spinner(f"Processing {uploaded_file.name}..."):
            cleaned, texts = deidentify(img)
            cleaned_rgb    = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(cleaned_rgb, caption="De-identified", use_container_width=True)

        if texts:
            st.success(f"Detected and removed: {', '.join(texts)}")
        else:
            st.info("No text detected.")

        # Download button
        cleaned_png = cv2.imencode(".png", cleaned)[1].tobytes()
        st.download_button(
            label="Download cleaned image",
            data=cleaned_png,
            file_name=f"{Path(uploaded_file.name).stem}_cleaned.png",
            mime="image/png"
        )

        st.divider()