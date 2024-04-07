import cv2
import time
import base64
import io
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
import requests

class ImageReader:
    def __init__(self):
        self.llm = Ollama(model="llava:34b-v1.6")
        self.prompt_template = ChatPromptTemplate.from_template(
            "Tell me what you see in the image, the closest person to the camera is the one you are talking to, address them as sir and speak in the second person. describe it as detailed as possible. You are a robot who can see the world..\n{image}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.runpod_url = "https://api.runpod.ai/v2/6tc9awj71ikxyh/run"
        self.api_key = "HEE27CRG3273S1FW0QM9N1TZJIMW7JO0UG9P3M95"


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
        return response

    def read_image_runpod(self, image_base64, input_text=None):
        url = "https://api.runpod.ai/v2/6tc9awj71ikxyh/runsync"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "input": {
                "image": image_base64,
                "prompt": input_text
            }
        }
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            try:
                result = response.json()
                if result["status"] == "IN_QUEUE":
                    request_id = result["id"]
                    print(f"Request {request_id} is in queue. Waiting for the result...")

                    # Periodically check the status of the request until it is completed
                    while True:
                        status_response = requests.post(f"{url}/status/{request_id}", headers=headers)

                        try:
                            status = status_response.json()["status"]
                        except requests.exceptions.JSONDecodeError:
                            print("Error: Unable to parse status response as JSON.")
                            print("Response content:", status_response.text)
                            break

                        if status == "COMPLETED":
                            output_response = requests.post(f"{url}/output/{request_id}", headers=headers)

                            try:
                                return output_response.json()["output"]
                            except requests.exceptions.JSONDecodeError:
                                print("Error: Unable to parse output response as JSON.")
                                print("Response content:", output_response.text)
                                break
                        elif status == "FAILED":
                            print("Error: Request failed.")
                            break
                        else:
                            time.sleep(1)  # Wait for 1 second before checking again
                else:
                    print("Error: Unexpected response status.")
            except KeyError:
                print("Error: Unexpected response format from RunPod server.")
                print("Response content:", response.text)
        else:
            print("Error: Request to RunPod server failed.")
            print("Status code:", response.status_code)
            print("Response content:", response.text)

        return None

    def read_camera_image(self, input_text=None):
        frame = self.capture_image()
        # Convert the captured image to a PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Convert the PIL image to a base64 string
        image_base64 = self.convert_to_base64(img)
        # Read the image using the AI model
        result = self.read_image_runpod(image_base64, input_text)
        if result is None:
            print("Failed to retrieve result from RunPod server.")
        return result

# Initialize the ImageReader
print("Starting program...")
image_reader = ImageReader()

# Read the text from the camera image using RunPod
print("Reading text from camera image using RunPod...")
result = image_reader.read_camera_image()
print("Text from the camera image:")
print('--------------------------')
print(result)
print("Program finished.")
