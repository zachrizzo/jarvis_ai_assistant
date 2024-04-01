# Jarvis AI Voice Assistant

Jarvis AI is a highly interactive voice assistant designed to perform a variety of tasks through voice commands. Utilizing advanced models such as OpenAI and Llama2, along with Porcupine for wake word detection, Jarvis can transcribe audio, execute functions, and even interact with the physical world through object detection.

## Features

- **Voice Activation**: Wake up Jarvis with a custom wake word.
- **Speech to Text**: Transcribe spoken commands into text.
- **Function Execution**: Perform actions based on the transcribed text.
- **Object Detection**: Visual recognition of objects in the environment.
- **Dynamic Response Generation**: Utilizes LLMs for generating responses and executing commands.
- And many more.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or later
- Access to OpenAI API and Porcupine API
- Install all required libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
   git clone https://github.com/zachrizzo/jarvis_ai_assistant

markdown
Copy code 2. Install the necessary Python packages in a conda env:
conda install -r requirements.txt

markdown
Copy code

## Configuration

1. Create a `.env` file in the project root directory.
2. Add your OpenAI and Porcupine API keys:
   OPENAI_API_KEY=<your_openai_api_key>
   PVPORCUPINE_API_KEY=<your_porcupine_api_key>

shell
Copy code

## Usage

To start the Jarvis AI voice assistant, run:

python main.py

markdown
Copy code

Speak your commands after the wake word is recognized. Jarvis will transcribe your speech, execute the appropriate function, and respond vocally.

## Customization

- Modify the `assistant_main.py` to add new functionalities or change the behavior of existing ones.
- Customize the wake word by changing the Porcupine keyword paths.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- OpenAI and llama2 for the GPT models
- Picovoice for the Porcupine wake word engine
