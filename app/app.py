import gradio as gr
from PIL import Image
from app.inference import predict_ecg

def run(image):
    result = predict_ecg(image)
    return (
        result["prediction"],
        result["probability"],
        result["trust"]
    )

demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Upload ECG Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Trust Decision")
    ],
    title="TrustECG",
    description=(
        "Confidence-aware ECG AI that predicts cardiac abnormality "
        "and explicitly abstains when predictions are unreliable."
    )
)

if __name__ == "__main__":
    demo.launch(share=True)

