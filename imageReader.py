import cv2
import time
import base64
import io
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
from langchain.tools import tool



class ImageReader:
    def __init__(self):
        self.llm = Ollama(model="llava:13b")
        self.prompt_template = ChatPromptTemplate.from_template(
            "Tell me what you see in the image, the closest person to the camera is the one you are talking to, address them as sir and speak in the second person. describe it as detailed as possible. You are a robot who can see the world..\n{image}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def capture_image(self):
        camera = cv2.VideoCapture(0)
        time.sleep(2)  # Let the camera adjust
        ret, frame = camera.read()
        if ret:
            print("Image captured.")
        else:
            print("Failed to capture image.")
        camera.release()
        return frame

    def convert_to_base64(self, image):
        """Convert a PIL image to a base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def read_image(self, image_base64, input_text=None):
        llm_with_image_context = self.llm.bind(images=[image_base64])
        response = llm_with_image_context.invoke(f"Describe the image in detail: here is the user prompt: {input_text}")
        print(f"LLM response: {response}")  # Log the LLM response
        return response

    @tool(return_direct=True)
    def read_camera_image(self, input_text=None):
        frame = self.capture_image()
        # Convert the captured image to a PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Convert the PIL image to a base64 string
        image_base64 = self.convert_to_base64(img)
        # Read the image using the AI model
        result = self.read_image(image_base64, input_text)
        if result is None:
            print("Failed to retrieve result.")
        print(f"Final result: {result}")  # Log the final result
        return {"result": result}
