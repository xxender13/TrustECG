import streamlit as st
from PIL import Image
from app.inference import predict_ecg

st.set_page_config(
    page_title="TrustECG",
    layout="centered"
)

st.title("🫀 TrustECG")
st.write(
    "Confidence-aware ECG AI that predicts abnormality "
    "and abstains when predictions are unreliable."
)

uploaded = st.file_uploader(
    "Upload an ECG image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded)

    st.image(image, caption="Uploaded ECG", use_column_width=True)

    with st.spinner("Running TrustECG inference..."):
        result = predict_ecg(image)

    st.subheader("Prediction")
    st.write(f"**Class:** {result['prediction']}")
    st.write(f"**Probability:** {result['probability']}")

    st.subheader("Trust Decision")
    st.write(result["trust"])
