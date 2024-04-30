import torch
import numpy as np
import whisper
import soundfile as sf
from openai import OpenAI
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
from pydub import AudioSegment
import simpleaudio as sa
import os
from record_audio import record_audio
from scipy.io.wavfile import write
import pvporcupine
import pyaudio
import struct
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from assistant_function_caller import FunctionCallerAI

class JarvisAI:
    def __init__(self, api_key, porcupine_api_key, model_name="llama", prompt=None):
        self.api_key = api_key
        self.porcupine_api_key = porcupine_api_key
        self.prompt = prompt
        self.openai_client = OpenAI(api_key=api_key)
        if model_name == "llama":
            self.llm = Ollama(model="llama3:8b-instruct-q6_K", temperature=0)
            #llama3:8b-instruct-q6_K
            #llama3:70b-instruct
        elif model_name == "openai":
            self.llm = ChatOpenAI(openai_api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.function_caller = FunctionCallerAI(llm=self.llm)
        self.prompt = hub.pull("hwchase17/react")

        self.tools = self.function_caller.tools
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps = True, max_iterations=1)

        self.conversation_history = []

    def transcribe_audio(self, audio_path):
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]

    def process_command(self, input_text):
        # self.conversation_history.append(("user", input_text))

        # Pass the input text to the ReAct agent
        response = self.agent_executor.invoke({"input": input_text})

        print(response)

        # Extract the tool output from the response
        tool_output = response["output"]

        # self.conversation_history.append(("assistant", tool_output))

        tool_output = response["output"]
        thoughts = response["thought"] if "thought" in response else []
        # Speak the tool output
        # self.generate_speech_openai(tool_output, output_path="output.wav")


        # End the chain by returning the tool output
        return tool_output, thoughts

    def conversation_summary(self):
        last_message = self.conversation_history[-1]
        template = "You use the last few thing in the convo to notify the user on what the ai has done. you will do this in one sentence and you reply in the first person \n\n{conversation_history}\n\nSummary: {summary}"
        prompt_template = ChatPromptTemplate.from_template(template)
        new_chain = LLMChain(llm=self.llm, prompt=prompt_template, output_parser=StrOutputParser())
        response = new_chain.invoke({"conversation_history": self.conversation_history, "summary": last_message})
        print('-----------------')
        print(response['text'])
        return response['text']

    def listen_for_wake_word(self, keyword_paths):
        porcupine = None
        pa = None
        audio_stream = None
        try:
            porcupine = pvporcupine.create(access_key=self.porcupine_api_key, keyword_paths=keyword_paths)
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    return
        finally:
            if porcupine is not None:
                porcupine.delete()
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()

    def generate_speech_openai(self, text, output_path="output.wav"):
        response = self.openai_client.audio.speech.create(model='tts-1', voice='onyx', input=text, response_format='mp3')
        temp_file = "temp_audio.mp3"
        response.stream_to_file(temp_file)
        audio = AudioSegment.from_file(temp_file, format="mp3")
        audio.export(output_path, format="wav")
        self.play_audio(output_path)
        os.remove(temp_file)
        return response

    def play_audio(self, file_path):
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def run(self):
        fs = 16000  # Sample rate
        keyword_path = '/Users/zachrizzo/Downloads/Jarvis_en_mac_v3_0_0/Jarvis_en_mac_v3_0_0.ppn'

        while True:
            print("Waiting for wake word 'Jarvis'...")
            self.listen_for_wake_word([keyword_path])
            print("Wake word detected. Listening for command...")
            filename = "command_audio.wav"
            audio_data = record_audio(fs=fs)
            if audio_data.size > 0:
                write(filename, fs, audio_data.astype(np.int16))

                print("Audio recorded and saved to:", filename)
                transcribed_text = self.transcribe_audio(filename)
                print("Transcribed Text:", transcribed_text)
                processed_text = self.process_command(transcribed_text)
                print("Processed Text:", processed_text)
                # final_response = self.conversation_summary()
                # Ensure final_response is a string
                # final_response_str = str(processed_text)
                # self.generate_speech_openai(final_response_str, output_path="output.wav")
            else:
                print("No audio recorded.")
