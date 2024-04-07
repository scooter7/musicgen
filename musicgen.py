import streamlit as st
import torch
from diffusers import AudioLDM2Pipeline
import numpy as np
from io import BytesIO

# Set up device and load pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
generator = torch.Generator(device)

def text2audio(text, negative_prompt, duration, guidance_scale, random_seed, n_candidates):
    generator.manual_seed(int(random_seed))
    waveforms = pipe(text, audio_length_in_s=duration, guidance_scale=guidance_scale,
                     num_inference_steps=200, negative_prompt=negative_prompt,
                     num_waveforms_per_prompt=n_candidates, generator=generator)["audios"]
    return waveforms[0]

# Streamlit UI
st.title("AudioLDM 2: A General Framework for Audio, Music, and Speech Generation")

# Input form
with st.form("text2audio_form"):
    text = st.text_input("Input text", value="The vibrant beat of Brazilian samba drums.")
    negative_prompt = st.text_input("Negative prompt", value="Low quality.")
    duration = st.slider("Duration (seconds)", 5, 15, value=10, step=2.5)
    guidance_scale = st.slider("Guidance scale", 0.0, 7.0, value=3.5, step=0.5)
    seed = st.number_input("Seed", value=45)
    n_candidates = st.slider("Number of waveforms to generate", 1, 5, value=3)
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    waveform = text2audio(text, negative_prompt, duration, guidance_scale, seed, n_candidates)
    waveform = np.asarray(waveform).astype(np.float32)

    audio_file = BytesIO()
    torch.save(waveform, audio_file)
    audio_file.seek(0)

    st.audio(audio_file, format="audio/wav")

# Additional tips or information can be added here similarly to how it's done in the Gradio app
