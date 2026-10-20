from typing import TypedDict, List,Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages:Annotated[List[AnyMessage],add_messages]
    current_query:str
    documents:List[str]
    plan:List[str]
    status:str
    final_answer:str