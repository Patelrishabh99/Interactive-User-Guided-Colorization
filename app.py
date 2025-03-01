import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import io
from streamlit_drawable_canvas import st_canvas


# Load the pre-trained colorization model
@st.cache_resource
def load_colorization_model():
    return load_model("model.h5", compile=False)


model = load_colorization_model()

# Streamlit App
st.title("🎨 Interactive User-Guided Colorization")
st.write("Upload a grayscale image and customize the colors interactively.")

# Upload Image
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded image to PIL Image
    img = Image.open(uploaded_file).convert("L")
    img_color = img.convert("RGB")  # Convert grayscale to RGB for editing
    st.image(img, caption="Uploaded Grayscale Image", use_column_width=True)

    # Convert PIL Image to NumPy array
    img_array = np.array(img.resize((512, 512))) / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Shape: (1, 512, 512, 1)

    # Perform auto-colorization
    if st.button("Auto-Colorize"):
        colorized_img = model.predict(img_array)[0]  # Output shape: (512, 512, 3)
        colorized_img = (colorized_img * 255).astype(np.uint8)
        colorized_pil = Image.fromarray(colorized_img)
        st.session_state["colorized_img"] = colorized_pil

    if "colorized_img" in st.session_state:
        st.image(st.session_state["colorized_img"], caption="Auto-Colorized Image", use_column_width=True)
        img_color = st.session_state["colorized_img"].copy()

    # User-defined color selection
    st.subheader("🎨 Customize Colors")
    color_picker = st.color_picker("Pick a color for selected regions", "#ff0000")

    # Convert HEX to RGB
    user_color = tuple(int(color_picker[i:i + 2], 16) for i in (1, 3, 5))

    # Allow drawing on the image
    st.subheader(" Selection Tool")
    st.write("Use your mouse to draw circles around the areas you want to color.")

    canvas_result = st_canvas(
        fill_color=color_picker + "ff",  # Ensure full opacity
        stroke_width=10,
        stroke_color=color_picker,
        background_image=img_color.copy(),
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode="freedraw",
        key="circle_canvas"
    )

    # Apply the drawn mask to the original image
    if canvas_result.image_data is not None:
        drawn_mask = Image.fromarray((canvas_result.image_data[:, :, :4]).astype(np.uint8), mode="RGBA")
        img_color = img_color.convert("RGBA")  # Ensure base image has an alpha channel
        img_color.paste(drawn_mask, (0, 0), drawn_mask)
        st.session_state["final_customized_img"] = img_color
        st.image(img_color, caption="Final Customized Image", use_column_width=True)

    # Save the final image
    if st.button("Download Final Image") and "final_customized_img" in st.session_state:
        buf = io.BytesIO()
        st.session_state["final_customized_img"].save(buf, format="PNG")
        st.download_button(label="Download", data=buf.getvalue(), file_name="colorized_image.png", mime="image/png")
