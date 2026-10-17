from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    `add_messages` is a LangGraph reducer that automatically appends new messages 
    to the list while keeping the conversation history intact.
    """
    messages: Annotated[List[BaseMessage], add_messages]

    # The Router's decision on which path to take
    router_decision: Optional[str]

    # Variables to hold intermediate tool results
    sql_query: Optional[str]
    sql_result: Optional[str]
    rag_context: Optional[str]

    # The final synthesized answer to return to the user
    final_answer: Optional[str]

    # Error handling and retry logic
    error: Optional[str]
    retry_count: int