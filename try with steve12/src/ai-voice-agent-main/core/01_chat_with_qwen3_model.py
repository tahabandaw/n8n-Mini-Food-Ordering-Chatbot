from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re

# Initialize Ollama model
llm = Ollama(model="qwen3:8b")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Build the chain
chain = prompt_template | llm

# In-memory chat history
chat_history = []

# Function to clean <think> blocks
def clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# Main chat loop
def chat():
    print("🤖 Qwen3 Chat (type 'exit' to quit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        cleaned = clean_response(response)
        print("Qwen3:", cleaned)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=cleaned))

if __name__ == "__main__":
    chat()