from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from imageReader import ImageReader

tools = [ ImageReader()]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")


# Choose the LLM to use
llm = OpenAI()

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=False, max_iterations=3)

# Define the conversation history
conversation_history = []

# Define the input text

input_text = "What do you see right now?"

# Pass the input text to the ReAct agent
response = agent_executor.invoke({"input": input_text})

print(response)

# Extract the tool output from the response
tool_output = response["output"]

# Print the tool output
print(tool_output)

# End the chain by returning the tool output
tool_output
# End of test.py


