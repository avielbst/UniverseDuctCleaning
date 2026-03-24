"""
DuctAI Copilot — LangChain agent using LangGraph's create_agent.

LangChain 1.2 uses create_agent() from langchain.agents (backed by LangGraph).
Memory is managed via MemorySaver keyed on thread_id — no manual history tracking needed.

Usage:
    from agent.agent import build_agent
    agent = build_agent()
    response = agent.invoke(
        {"messages": [HumanMessage(content="What should I quote for Boca Raton?")]},
        config={"configurable": {"thread_id": "owner-session"}},
    )
    print(response["messages"][-1].content)
"""

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

from agent.system_prompt import build_system_prompt
from agent.tools import upsell_tool, pricing_tool, sql_tool


def build_agent():
    """
    Build and return the DuctAI Copilot agent.

    Returns a LangGraph CompiledStateGraph. Invoke it with:
        agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": session_id}},
        )
    The response["messages"][-1].content contains the final reply.
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    tools = [upsell_tool, pricing_tool, sql_tool]
    checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
    )
    return agent
