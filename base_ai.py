import requests
from openai import OpenAI
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI

class BaseAI:
    def __init__(self, model_name="llama2", api_key=None, runpod_endpoint=None, runpod_token=None):
        self.api_key = api_key
        self.runpod_endpoint = runpod_endpoint
        self.runpod_token = runpod_token
        self.model_name = model_name
        self.llm = self.initialize_llm(model_name, api_key)

    def initialize_llm(self, model_name, api_key):
        if model_name.startswith("llama"):
            return Ollama(model=model_name)
        elif model_name == "openai":
            return ChatOpenAI(openai_api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def process(self, data, use_runpod=False):
        if use_runpod:
            return self.llm_runpod(data)
        else:
            return self.llm_locally(data)

    def read_image(self, image_base64, input_text=None, use_runpod=False):
        if use_runpod:
            # Prepare data for RunPod including image and input text
            data = {"image": image_base64, "input_text": input_text}
            return self.llm_runpod(data)
        else:
            if isinstance(self.llm, Ollama):
                llm_with_image_context = self.llm.bind(images=[image_base64])
                response = llm_with_image_context.invoke(f"Describe the image in detail: here is the user prompt: {input_text}")
                return response
            else:
                raise NotImplementedError("Local image reading is not implemented for this LLM type.")

    def llm_runpod(self, data):
        if not self.runpod_endpoint or not self.runpod_token:
            raise ValueError("RunPod endpoint or token is not configured.")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.runpod_token}',
        }
        payload = self.prepare_runpod_payload(data)
        response = requests.post(self.runpod_endpoint, json=payload, headers=headers)
        return response.json()

    def llm_locally(self, data):
        # Local LLM processing (adapt based on your use case)
        if isinstance(self.llm, Ollama):
            return self.llm.invoke(data)
        elif isinstance(self.llm, ChatOpenAI):
            return self.llm.create_completion(data)

    def prepare_runpod_payload(self, data):
        # Prepare your data correctly for RunPod's expected format
        return {"input": data}

# # Usage example
# if __name__ == "__main__":
#     api_key = "your_api_key_here"
#     runpod_endpoint = "https://api.runpod.ai/v2/your_endpoint_id/run"
#     runpod_token = "your_runpod_token_here"
#     model_name = "llama2"  # Or "openai" for ChatGPT

#     base_ai = BaseAI(model_name=model_name, api_key=api_key, runpod_endpoint=runpod_endpoint, runpod_token=runpod_token)

#     image_base64 = "your_base64_encoded_image_here"
#     input_text = "Optional input text"

#     # Process image description locally or on RunPod
#     description = base_ai.read_image(image_base64, input_text=input_text, use_runpod=False)  # Set use_runpod=True to use RunPod
#     print(description)
