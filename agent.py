
"""Simple interactive AI agent powered by Ollama."""
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
MODEL = "llama3.2"
SYSTEM_PROMPT = "You are a helpful, friendly AI assistant. Be concise and clear."
EXIT_COMMANDS = {"quit", "exit", "bye", "q"}

def main() -> None:
    llm = ChatOllama(model=MODEL, temperature=0.7)
    history: list = [SystemMessage(content=SYSTEM_PROMPT)]
    print(f"AI Agent (Ollama / {MODEL})")
    print("Type your message and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break
        history.append(HumanMessage(content=user_input))
        try:
            response = llm.invoke(history)
        except Exception as exc:
            print(f"\nError: {exc}")
            print("Make sure Ollama is running and the model is pulled (ollama pull llama3.2).\n")
            history.pop()
            continue
        reply = response.content.strip()
        history.append(AIMessage(content=reply))
        print(f"\nAgent: {reply}\n")

if __name__ == "__main__":
    main()
