"""
Router Node: Classifies user intent and decides which tool(s) to use.
"""
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from models import get_llm

def route_question(state: dict) -> Literal["sql", "rag", "both"]:
    """
    Analyzes the user question and routes to the appropriate tool.
    Returns: "sql", "rag", or "both"
    """
    router_prompt = ChatPromptTemplate.from_template(
        """You are an expert at routing user questions to the right data source.
        
        You have access to:
        1. SQL Database: Contains bank transaction data (customers, accounts, transactions)
           - Use for: specific numbers, counts, averages, transaction details, account balances
           - Examples: "How many transactions over $5000?", "What is the total balance of account 123?"
        
        2. Vector Database: Contains bank fraud policies and procedures
           - Use for: rules, policies, SOPs, definitions, procedures
           - Examples: "What is the policy for late fees?", "What are the Red Flag Indicators?"
        
        3. Both: If the question requires data from both sources
           - Examples: "Calculate the total late fees for customer X and tell me the policy for waiving them"
        
        User Question: {question}
        
        Respond with ONLY one word: "sql", "rag", or "both"
        """
    )

    llm = get_llm()
    chain = router_prompt | llm
    response = chain.invoke({"question": state["messages"][-1].content})

    decision = response.content.strip().lower()

    # Ensure valid output
    if decision not in ["sql", "rag", "both"]:
        decision = "rag"    # Default fallback

    return decision