
from assistant_main import JarvisAI
from dotenv import load_dotenv
import os
load_dotenv()

OpenAI_api_key=os.getenv("OPENAI_API_KEY")
porcupine_api_key=os.getenv("PVPORCUPINE_API_KEY")




# Main loop for processing audio input and generating responses
if __name__ == "__main__":
    model_name = 'openai'  # Specify the desired model name here
    jarvis = JarvisAI(OpenAI_api_key,porcupine_api_key, model_name)
    jarvis.run()

