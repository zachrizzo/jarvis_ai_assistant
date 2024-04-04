from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from img_rec import ObjectDetector
import random
import string
import os
import importlib.util
import re
from code_writter import CodeGenerator
from imageReader import ImageReader

class FunctionCallerAI:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template(
            "Based on the following input, decide which function to call, and you will not call a function that is already running:\n\n"
            "Input: {input}\n\n"
            "Currently running functions: {running_functions}\n\n"
            "Available functions:\n{functions_list}\n\n"
            "If the input asks about detected objects, call 'query objects in object detection'.\n"
            "If no specific function is needed, respond with 'no function needed'.\n\n"
            "Function to call: "
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.object_detector = ObjectDetector()
        self.code_generator = CodeGenerator(newLLM=self.llm)
        self.image_reader = ImageReader()

        # Track running functions
        self.running_functions = set()

        # Define functions the AI can call
        self.functions = {
            "no function needed": {
                "func": self.no_function_needed,
                "description": "Use this when no specific action is required.",
                'function_type': 'single'
            },
            # "what the ai sees": {
            #     "func": self.object_detector.get_latest_results,
            #     "description": "Queries and describes the latest detected objects, showing what the AI sees.",
            #     'function_type': 'single'
            # },
            # "object detection": {
            #     "func": self.object_detector.stream_and_detect,
            #     "description": "Starts object detection to identify objects in the surrounding environment.",
            #     'function_type': 'continuous'
            # },
            "stop object detection": {
                "func": self.object_detector.stop_object_detection,
                "description": "Stops the ongoing object detection process.",
                'function_type': 'single'

            },
            'create and execute new function': {
                'func': self.code_generator.run,
                'description': 'Creates a new Python file that code can be written to and executed by the ai.',
                'function_type': 'single'
            },
            'tell me what you see right now': {
                'func': self.image_reader.read_camera_image,
                'description': 'Captures an image from the camera and describes the objects in the image.',
                'function_type': 'single'
            }

        }

    def no_function_needed(self, input_text):
        return "No function needed."




    def add_function(self, name, func, description):
        """Add a callable function with its description to the AI's repertoire."""
        self.functions[name] = {"func": func, "description": description}

    def getReadableFunctionList(self):
        """Return a human-readable list of available functions."""
        return "\n".join([f"- {name}: {desc['description']}" for name, desc in self.functions.items()])

    def getRunningFunctions(self):
        """Return a list of currently running functions."""
        return ", ".join(self.running_functions) if self.running_functions else "None"

    def create_and_execute_new_function(self, input_text):
        """Generates Python code based on input text, writes to a new Python file, and executes all top-level functions."""
        print('Generating code for:', input_text)

        # Generate Python code using LLM and LangChain
        code_prompt = ChatPromptTemplate.from_template("Generate Python code based on the following input:\n\n{input}\n\n```python\n")
        code_chain = LLMChain(llm=self.llm, prompt=code_prompt)
        generated_code = code_chain.invoke({"input": input_text})['text']

        print('Generated code:', generated_code)

        if not generated_code.strip():
            print("Failed to generate Python code.")
            return

        # Extract code if wrapped in ```
        if '```' in generated_code:
            generated_code = generated_code.split('```')[1]

        # Create a directory for user functions if it doesn't exist
        directory = "user_functions"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write the generated Python code to a new file
        filename = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.py'
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            file.write(generated_code)

        # Dynamically import the created file as a module
        spec = importlib.util.spec_from_file_location("dynamic_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all top-level functions in the module and execute them
        for item in dir(module):
            if not item.startswith("__"):
                possible_function = getattr(module, item)
                if callable(possible_function):
                    print(f"Executing {item}:")
                    try:
                        result = possible_function()
                        print(f"Result from {item}: {result}")
                    except Exception as e:
                        print(f"Error executing {item}: {e}")

    def decide_and_call(self, input_text):
        # Prepare input for the LLM
        running_functions_list = self.getRunningFunctions()
        functions_list = self.getReadableFunctionList()
        llm_input = f'{input_text}\nhere are the functions you can call:\n{functions_list}'

        suggested_action = self.chain.invoke({
            'input': llm_input,
            'running_functions': running_functions_list,
            'functions_list': functions_list
        })

        print(f"Suggested action: {suggested_action}")

        # Extract the suggested action text
        suggested_func_name = suggested_action['text'].strip().lower()

        # Dynamically decide and call the function
        for func_name, details in self.functions.items():
            if func_name.lower() in suggested_func_name:
                if func_name in self.running_functions:
                    if "stop" in func_name.lower():  # Handling stop requests
                        self.running_functions.remove(func_name)
                        result = details["func"]()
                        return f"Stopped function '{func_name}', result: {result}"
                    else:
                        return f"Function '{func_name}' is already running."
                else:
                    if "stop" not in func_name.lower():  # Avoid stopping non-running functions
                        if details['function_type'] == 'continuous':
                            self.running_functions.add(func_name)
                        result = details["func"](input_text)
                        return f"Started function '{func_name}', result: {result}"
                    else:
                        return f"Function '{func_name}' is not running, so it cannot be stopped."

        return "No appropriate function found for the given input."
