import base64
import json
# Read the image file
with open("img.jpeg", "rb") as image_file:
    image_bytes = image_file.read()

# Encode the image bytes as base64
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# Create the test_input.json file
with open("test_input.json", "w") as test_input_file:
    test_input = {
        "input": {
            "image": image_base64,
            "text": "Optional input text"
        }
    }
    test_input_file.write(json.dumps(test_input))
print
print("test_input.json file created successfully.")
