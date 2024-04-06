# Use Python 3.11.8 as the base image
FROM python:3.11.8

# Set the working directory in the container to /ai_api
WORKDIR /ai_api

# Install curl which is required to install Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*


# Copy the requirements.txt file first to leverage Docker cache
COPY ./ai_api_docker_runpod/requirements_pip.txt ./

# Install any dependencies in the requirements.txt
RUN pip install --no-cache-dir -r requirements_pip.txt

# Now copy the rest of your application's source code from the local ai_api_docker_runpod directory to the container's work directory
COPY ./ai_api_docker_runpod ./

# Install Ollama using its installation script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server in the background
RUN ollama serve & sleep 3 && ollama run llava:34b-v1.6


# Start the Ollama app
CMD ["python","-u", "main.py"]
