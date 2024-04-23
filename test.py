from langchain_community.llms.ollama import Ollama

llm = Ollama(model="llama3:8b-instruct-q6_K")


# llm.invoke("Tell me a joke",stop=['<|eot_id|>'])


query = "Tell me a joke"

for chunks in llm.stream(query, stop=['<|eot_id|>']):
    print(chunks)
