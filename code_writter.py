from langchain_community.llms.ollama import Ollama
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
import uuid
import re
import subprocess
import sys


class CodeGenerator:
    def __init__(self, newLLM = '' ):
        self.llm = ''
        if newLLM != '':
            self.llm = newLLM
        else:
            self.llm = Ollama(model="llama2:70b-chat-q5_K_M")


    def generate_code(self,prompt) -> str:
        prompt_template = ChatPromptTemplate.from_template("{prompt}\nGenerate Python code to solve the above problem:")
        code = LLMChain(llm=self.llm, prompt=prompt_template).invoke(input={'prompt': prompt})
        return self.extract_code(code.get('text'))

    def extract_code(self, response: str) -> str:
        code_block_pattern = r"(```python[\s\S]*?```|```[\s\S]*?```)"
        code_blocks = re.findall(code_block_pattern, response)

        if code_blocks:
            raw_code = re.sub(r"(```python|```)", "", code_blocks[0]).strip()
            return raw_code
        else:
            if response.strip() == "":
                raise Exception("No code found in the response.")
            return response.strip()

    def fix_code(self, code: str, file_path: str) -> str:
        prompt_template =ChatPromptTemplate.from_template("Here is the generated Python code:\n\n{code}\n\nTest the code and fix any bugs. If there are any missing dependencies or undefined variables, please modify the code accordingly. Provide only the fixed Python code without any explanations or comments.")
        llm_chain = LLMChain(llm=self.llm, prompt= prompt_template)
        fixed_code = llm_chain.invoke(input={'code':code})
        fixed_code = self.extract_code(fixed_code)
        with open(file_path, "w") as file:
            file.write(fixed_code)
        return fixed_code

    def save_code(self, code: str) -> str:
        if not os.path.exists("user_functions"):
            os.makedirs("user_functions")
        file_name = f"{uuid.uuid4().hex}.py"
        file_path = os.path.join("user_functions", file_name)
        with open(file_path, "w") as file:
            file.write(code)
        return file_path

    def run_code(self, file_path: str) -> str:
        while True:
            try:
                with open(file_path, "r") as file:
                    code = file.read()
                exec(code, {})
                return "Code executed successfully."
            except ModuleNotFoundError as e:
                missing_module = str(e).split("'")[1]
                print(f"Missing module: {missing_module}")
                self.install_package(missing_module)
                fixed_code = self.fix_code(code, file_path)
                print(f"\nFixed code:\n{fixed_code}")
            except Exception as e:
                error_message = f"Error: {str(e)}"
                print(error_message)
                fixed_code = self.fix_code(code, file_path)
                print(f"\nFixed code:\n{fixed_code}")

    def install_package(self, package_name: str) -> None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' installed successfully.")
            print('---------------------------------------')
            print('installed package', package_name)
            print('---------------------------------------')
            #save a list of installed packages in a file
            with open('installed_packages.txt', 'a') as file:
                file.write(f"{package_name}\n")

        except subprocess.CalledProcessError as e:
            print(f"Error installing package '{package_name}': {str(e)}")

    def run(self, prompt) -> str:
        code = self.generate_code(prompt)
        # print(f"Generated code:\n{code}")
        print('---------------------------------------')
        print('generating code')
        print('---------------------------------------')
        file_path = self.save_code(code)
        result = self.run_code(file_path)
        print(f"Execution result: {result}")
        return code

# # Example usage
# if __name__ == "__main__":
#     # User prompt for code generation
#     prompt = "Write a Python function to find to facial recognition from a live feed from mac book webcam."

#     # Create an instance of the CodeGenerator class
#     code_generator = CodeGenerator(prompt)

#     # Run the code generation and bug fixing process
#     final_code = code_generator.run()
#     print("\nFinal code:")
#     print(final_code)
