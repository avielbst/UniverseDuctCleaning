"""
DuctAI Copilot — interactive CLI.

Run:
    python -m agent.cli

Type your question and press Enter. Type 'exit' or Ctrl-C to quit.
History is maintained across turns in the same session via LangGraph MemorySaver.
"""
import sys

from langchain_core.messages import HumanMessage

from agent.agent import build_agent

SESSION_ID = "cli"


def main():
    print("DuctAI Copilot — type your question, or 'exit' to quit.\n")
    agent = build_agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            sys.exit(0)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": SESSION_ID}},
            )
            reply = result["messages"][-1].content
            print(f"\nAgent: {reply}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
