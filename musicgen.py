from audiocraft.models import MusicGen
import streamlit as st
import torch
import numpy as np
from pydub import AudioSegment
from io import BytesIO

@st.experimental_singleton
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

@st.experimental_memo
def generate_music_tensors(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )
    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]

def tensor_to_mp3_bytes(tensor, sample_rate=32000):
    tensor = tensor.detach().cpu()
    numpy_audio = tensor.numpy()
    if numpy_audio.dtype == np.float32:
        numpy_audio = (numpy_audio * 32767.0).astype(np.int16)
    audio_segment = AudioSegment(
        numpy_audio.tobytes(), 
        frame_rate=sample_rate,
        sample_width=numpy_audio.dtype.itemsize, 
        channels=1
    )
    buffer = BytesIO()
    audio_segment.export(buffer, format="mp3")
    return buffer.getvalue()

st.set_page_config(page_icon="musical_note", page_title="Music Gen")

st.title("Text to Music Generator🎵")

with st.expander("See explanation"):
    st.write("Music Generator app built using Meta's Audiocraft library. We are using Music Gen Small model.")

text_area = st.text_area("Enter your description")
time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

if text_area and time_slider:
    st.json({
        'Your Description': text_area,
        'Selected Time Duration (in Seconds)': time_slider
    })

    st.subheader("Generated Music")
    music_tensors = generate_music_tensors(text_area, time_slider)
    audio_bytes = tensor_to_mp3_bytes(music_tensors)
    st.audio(audio_bytes, format='audio/mp3')
    st.download_button("Download Audio", audio_bytes, "generated_music.mp3", "audio/mp3")
