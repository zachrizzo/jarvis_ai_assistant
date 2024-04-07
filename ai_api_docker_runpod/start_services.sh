#!/bin/bash
ollama serve & sleep 3 && ollama run llava:34b-v1.6

# Start your main Python application
exec python -u main.py
