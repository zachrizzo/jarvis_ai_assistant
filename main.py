
from assistant_main import JarvisAI
from dotenv import load_dotenv
import os
load_dotenv()

OpenAI_api_key=os.getenv("OPENAI_API_KEY")
porcupine_api_key=os.getenv("PVPORCUPINE_API_KEY")




# Main loop for processing audio input and generating responses
if __name__ == "__main__":
    #possible models from ollama

    #llaama2:70b-chat-q5_K_M
    #llama2

    #--------Vision:
    #llava:34b-v1.6
    #llava:7b-v1.6

    model_name = 'openai'  # Specify the desired model name here
    jarvis = JarvisAI(OpenAI_api_key,porcupine_api_key, model_name)
    jarvis.run()

