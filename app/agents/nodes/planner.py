from typing import Literal
from app.gateway.client import  get_langchain_llm
import logfire
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.config import settings


class PlannerDecision(BaseModel):
    """Classify whether the latest message needs document retrieval."""

    category: Literal["TECHNICAL", "CONVERSATIONAL"] = Field(
        description=(
            "TECHNICAL for questions requiring technical documentation. "
            "CONVERSATIONAL for greetings, thanks, or questions answerable "
            "from conversation history."
        )
    )


llm = get_langchain_llm(feature="planner")
    

planner_llm = llm.with_structured_output(
    PlannerDecision,
    method="function_calling",
)

SYSTEM_PROMPT = """
You classify messages for an enterprise documentation assistant.

Choose TECHNICAL when the latest message requires searching technical
documentation, including follow-up questions about technical topics.

Choose CONVERSATIONAL for greetings, thanks, casual conversation, or
questions that can be answered entirely from conversation history.

Use the conversation history to understand the latest message.
"""


def planner_node(state: AgentState) -> dict:
    messages = state["messages"]

    if not messages:
        raise ValueError("The planner requires at least one message.")

    with logfire.span("Classify user query") as span:
        decision = planner_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *messages,
            ]
        )

        span.set_attribute("decision", decision.category)

        # Fail explicitly if no validated decision was returned.
        if not isinstance(decision, PlannerDecision):
            raise ValueError("The planner did not return a valid decision.")

        if decision.category == "CONVERSATIONAL":
            return {
                "current_query": "CONVERSATIONAL",
                "plan": ["Respond using conversation history."],
                "status": "Preparing conversational response.",
            }

        latest_content = messages[-1].content
        if not isinstance(latest_content, str):
            raise ValueError("The planner currently requires a text query.")

        return {
            "current_query": latest_content,
            "plan": [
                "Retrieve relevant technical documentation.",
                "Generate an answer using the retrieved context.",
            ],
            "status": "Searching technical documentation.",
        }