
import streamlit as st
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import tempfile
import math
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

@st.cache_resource
def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()
transform = T.Compose([T.ToTensor()])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def get_frame_intervals(duration):
    if duration <= 60:
        interval = 2
    elif duration <= 500:
        interval = 5
    else:
        interval = 10

    max_frames = 100
    total_frames = duration // interval
    if total_frames > max_frames:
        interval = math.ceil(duration / max_frames)
    return interval

def detect_objects(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions['labels'].numpy() if i < len(COCO_INSTANCE_CATEGORY_NAMES)]
    return list(set(labels))

st.title("🎯 YouTube Product Tag Extractor")
st.write("Paste a YouTube link and get smart product tags using AI!")

link = st.text_input("Paste YouTube video link:")

if link:
    try:
        yt = YouTube(link)
        st.video(link)
        with st.spinner("Downloading and processing video..."):
            temp_dir = tempfile.mkdtemp()
            path = yt.streams.filter(file_extension='mp4').first().download(output_path=temp_dir)
            clip = VideoFileClip(path)
            duration = int(clip.duration)
            interval = get_frame_intervals(duration)

            tags = set()
            for t in range(0, duration, interval):
                frame_path = os.path.join(temp_dir, f"frame_{t}.png")
                clip.save_frame(frame_path, t)
                image = Image.open(frame_path).convert("RGB")
                labels = detect_objects(image)
                tags.update(labels)

            st.success(f"✅ Done! Detected {len(tags)} unique tags.")
            st.write(sorted(tags))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
