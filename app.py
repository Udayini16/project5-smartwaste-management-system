import streamlit as st
from roboflow import Roboflow
import supervision as sv
from supervision.detection.annotate import BoxAnnotator
from supervision.draw.color import ColorPalette
import cv2
import numpy as np
from PIL import Image
import tempfile

# App title
st.title("♻️ Smart Waste Detection using Roboflow")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Roboflow prediction
    rf = Roboflow(api_key="NfDIqRxz2n6kOqEmL8If")
    project = rf.workspace().project("smart-waste-management-h5yif-mwcpw")
    model = project.version(1).model
    result = model.predict(image_path, confidence=0.4, overlap=0.3).json()

    predictions = result["predictions"]

    if not predictions:
        st.warning("No objects detected.")
    else:
        # Extract detection data
        xyxy =[]
        confidences=[]
        class_ids=[]
        labels = []

        for pred in predictions:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)

            xyxy.append([x1, y1, x2, y2])
            confidences.append(pred["confidence"])
            class_ids.append(pred["class_id"])
            labels.append(pred["class"])

        # Create Detections object
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        # Annotate image
        annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2, text_scale=0.5)
        annotated_img = annotator.annotate(scene=image_np.copy(), detections=detections, labels=labels)

        # Display
        st.image(annotated_img, caption=" Annotated Output", use_column_width=True)

        # Download option
        bgr_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".jpg", bgr_annotated)
        if success:
            st.download_button("📥 Download Annotated Image", data=encoded_image.tobytes(), file_name="annotated_output.jpg", mime="image/jpeg")

        # Display detected classes
        st.subheader("📦 Detected Classes")
        for cls in sorted(set(labels)):
            st.write(f"• {cls}")
