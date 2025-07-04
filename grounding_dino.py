import os
import uuid
import time
import zipfile
import io
import requests
import streamlit as st

nvai_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"
nvai_polling_url_base = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
UPLOAD_ASSET_TIMEOUT = 300
DELAY_BTW_RETRIES = 2
MAX_RETRIES = 10

NVIDIA_API_KEY = os.getenv("NVIDIA_PERSONAL_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

header_auth = f"Bearer {NVIDIA_API_KEY}"

def upload_asset_to_nvidia(video_bytes, description="Input Video"):
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "video/mp4",
    }

    payload = {"contentType": "video/mp4", "description": description}
    response = requests.post(assets_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(asset_url, data=video_bytes, headers=s3_headers, timeout=UPLOAD_ASSET_TIMEOUT)
    response.raise_for_status()

    return str(uuid.UUID(asset_id))

def display_zip_contents(zip_bytes):
    if isinstance(zip_bytes, io.BytesIO):
        zip_bytes = zip_bytes.getvalue()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        st.write("### Zip Archive Contents")
        st.write(z.namelist())
        video_files = next((f for f in z.namelist() if f.endswith('.mp4')),None)
        if video_files is None:
            st.error("No video files found in the zip archive.")
            return
        video_bytes = z.read(video_files)
        st.video(video_bytes)

def run_grounding_dino_inference(prompt, video_bytes):
    asset_id = upload_asset_to_nvidia(video_bytes)

    inputs = {
        "model": "Grounding-Dino",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "media_url", "media_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}}
                ]
            }
        ],
        "threshold": 0.3
    }

    headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_id,
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
        "Authorization": header_auth,
    }

    print("Sending request to Grounding DINO endpoint...")
    response = requests.post(nvai_url, headers=headers, json=inputs)

    if response.status_code == 200:
        display_zip_contents(response.content)
        return

    elif response.status_code == 202:
        print("Pending evaluation... Polling for results.")
        req_id = response.headers.get("NVCF-REQID")
        polling_url = nvai_polling_url_base + req_id

        for _ in range(MAX_RETRIES):
            time.sleep(DELAY_BTW_RETRIES)
            poll_response = requests.get(polling_url, headers={"accept": "application/json", "Authorization": header_auth})
            if poll_response.status_code == 200:
                display_zip_contents(io.BytesIO(poll_response.content))
                return
            elif poll_response.status_code == 202:
                print(" Still processing...")
            else:
                print(f" Unexpected status: {poll_response.status_code}")
                break

        print(" Timeout: Result not ready in time.")
    else:
        print(f" Failed: {response.status_code}, {response.text}")
