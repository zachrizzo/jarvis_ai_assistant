from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
import runpod

class ImageReader:
    def __init__(self):
        self.llm = Ollama(model="llava:34b-v1.6")
        self.prompt_template = ChatPromptTemplate.from_template(
            "Tell me what you see in the image, the closest person to the camera is the one you are talking to, address them as sir and speak in the second person. describe it as detailed as possible. You are a robot who can see the world..\n{image}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def read_image(self, image_base64, input_text=None):
        llm_with_image_context = self.llm.bind(images=[image_base64])
        response = llm_with_image_context.invoke(f"Describe the image in detail: here is the user prompt: {input_text}")
        return response

def handler(job):
    job_input = job["input"]
    image_base64 = job_input["image"]
    input_text = job_input.get("text", None)

    try:
        # Create an instance of ImageReader
        image_reader = ImageReader()

        # Call the read_image method with the base64-encoded image string
        response = image_reader.read_image(image_base64, input_text)

        return {"output": response}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
