import streamlit as st
import os
from uuid import uuid4
from pathlib import Path
from grounding_dino import run_grounding_dino_inference

st.set_page_config(
    page_title="Grounding DINO Video Labeling",
    layout="wide"
)

st.title("Grounding DINO Video Labeling App")
st.markdown("Upload a video : ")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
prompt = st.text_input("Enter your prompt : ")

if uploaded_file is not None:
    video_id = str(uuid4())
    video_path = Path("uploads") / f"{video_id}.mp4"
    output_dir = Path("outputs")

    Path("uploads").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video uploaded successfully: {video_path}")

    if st.button("Run Grounding DINO") and uploaded_file:
        video_bytes = uploaded_file.read()   
        run_grounding_dino_inference(prompt, video_bytes)
        st.success("Processing complete!")
        if isinstance(video_path, Path):
            st.video(str(video_path))
        else:
            st.video(video_path)
        output_zip = output_dir / f"{video_id}.zip"
        if output_zip.exists():
            with open(output_zip, "rb") as f:
                st.download_button(
                    label="Download Output",
                    data=f,
                    file_name=output_zip.name,
                    mime="application/zip"
                )
        else:
            st.error("Output file not found. Please check the processing step.")