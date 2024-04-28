
from assistant_main import JarvisAI
from dotenv import load_dotenv
import os
import streamlit as st
import numpy as np

load_dotenv()

OpenAI_api_key=os.getenv("OPENAI_API_KEY")
porcupine_api_key=os.getenv("PVPORCUPINE_API_KEY")
# set up API key from .env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def process_text_input(jarvis, text_input):
    processed_text = jarvis.process_command(text_input)
    st.write("Processed Text:", processed_text)
    st.audio("output.wav", format="audio/wav")

def main():
    st.title("Jarvis AI")

    model_name = st.selectbox("Select Model", ["llama", "openai"])
    jarvis = JarvisAI(OpenAI_api_key, porcupine_api_key, model_name)

    text_input = st.text_input("Enter your command")
    if st.button("Process Text"):
        if text_input:
            process_text_input(jarvis, text_input)
        else:
            st.write("Please enter a command.")

    if st.button("Start Listening"):
        st.write("Waiting for wake word 'Jarvis'...")
        jarvis.listen_for_wake_word(["/Users/zachrizzo/Downloads/Jarvis_en_mac_v3_0_0/Jarvis_en_mac_v3_0_0.ppn"])
        st.write("Wake word detected. Listening for command...")

        filename = "command_audio.wav"
        fs = 16000
        audio_data = jarvis.record_audio(fs=fs)

        if audio_data.size > 0:
            jarvis.write(filename, fs, audio_data.astype(np.int16))
            st.write("Audio recorded and saved to:", filename)

            transcribed_text = jarvis.transcribe_audio(filename)
            st.write("Transcribed Text:", transcribed_text)

            process_text_input(jarvis, transcribed_text)
        else:
            st.write("No audio recorded.")



# Main loop for processing audio input and generating responses
if __name__ == "__main__":
    #possible models from ollama

    #llaama2:70b-chat-q5_K_M
    #llama2

    #--------Vision:
    #llava:34b-v1.6
    #llava:7b-v1.6

   main()

