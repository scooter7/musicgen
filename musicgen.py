from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64
import librosa  # For MP3 conversion if necessary

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, genre, mood, tempo, duration: int):
    description = f"{genre}, {mood}, {tempo} BPM, {description}"
    print("Description: ", description)
    print("Duration: ", duration)
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

def save_audio(samples: torch.Tensor):
    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    audio_path = os.path.join(save_path, "audio_0.mp3")  # Assuming single output for simplicity

    torchaudio.save(audio_path, samples.detach().cpu(), sample_rate)
    # If necessary, use librosa or another library to convert WAV to MP3 here
    return audio_path

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("AI Music Generator")

    genre = st.selectbox('Select Genre', ['Rock', 'Jazz', 'Hip Hop', 'Electronic'])
    mood = st.selectbox('Select Mood', ['Happy', 'Sad', 'Energetic', 'Calm'])
    tempo = st.slider('Tempo (BPM)', 60, 180, 120)
    description = st.text_area("Enter your description:")
    duration = st.slider("Duration (In Seconds)", 0, 20, 10)

    if st.button("Generate Music"):
        music_tensor = generate_music_tensors(description, genre, mood, tempo, duration)
        audio_path = save_audio(music_tensor)
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        st.markdown(get_binary_file_downloader_html(audio_path, 'MP3'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
