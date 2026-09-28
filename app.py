"""Interactive demo: draw a digit, get a live prediction.

Run:  streamlit run app.py
Requires digit_model.keras (run train_model.py first).
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "digit_model.keras"
CANVAS_SIZE = 280
MNIST_SIZE = 28
INNER_SIZE = 20  # MNIST digits occupy a centered 20x20 box

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image_data: np.ndarray) -> np.ndarray:
    """Canvas RGBA -> (1, 28, 28, 1) tensor, normalized like MNIST.

    Crops to the drawn digit's bounding box, scales it into a 20x20 box
    (preserving aspect ratio), and centers it on a 28x28 canvas — the same
    format the model was trained on.
    """
    gray = image_data[:, :, :3].mean(axis=2).astype(np.uint8)

    # Bounding box of the drawn pixels
    rows = np.any(gray > 20, axis=1)
    cols = np.any(gray > 20, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((1, MNIST_SIZE, MNIST_SIZE, 1), dtype="float32")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    digit = gray[rmin : rmax + 1, cmin : cmax + 1]

    # Scale longest side to 20px, keep aspect ratio
    h, w = digit.shape
    scale = INNER_SIZE / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = np.array(
        Image.fromarray(digit).resize((new_w, new_h), Image.LANCZOS)
    )

    # Paste centered into 28x28
    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
    y = (MNIST_SIZE - new_h) // 2
    x = (MNIST_SIZE - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = digit

    return (canvas / 255.0).astype("float32").reshape(1, MNIST_SIZE, MNIST_SIZE, 1)


def main():
    st.title("✏️ Handwritten Digit Recognizer")
    st.write("Draw a digit (0–9) below — a CNN trained on MNIST predicts it live.")

    try:
        model = load_model()
    except Exception:
        st.error("Model not found. Run `python3 train_model.py` first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Draw here")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        st.caption("Tip: draw big and centered for best results.")

    with col2:
        st.subheader("Prediction")
        if canvas_result.image_data is not None:
            tensor = preprocess(canvas_result.image_data)

            if tensor.sum() == 0:
                st.info("Draw a digit to see the prediction.")
            else:
                probs = model.predict(tensor, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred]) * 100

                st.markdown(
                    f"<p style='font-size:72px; font-weight:700; text-align:center; "
                    f"margin:0'>{pred}</p>"
                    f"<p style='font-size:18px; text-align:center; color:gray'>"
                    f"{conf:.1f}% confident</p>",
                    unsafe_allow_html=True,
                )

                chart_data = {str(i): float(probs[i]) for i in range(10)}
                st.bar_chart(chart_data)

                with st.expander("What the model sees (28×28)"):
                    st.image(
                        tensor[0, :, :, 0],
                        width=140,
                        clamp=True,
                    )

    st.divider()
    st.caption(
        "CNN: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(64) → Softmax(10) "
        "· Trained on 60k MNIST images · ~99% test accuracy"
    )


if __name__ == "__main__":
    main()
