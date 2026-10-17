"""
LangGraph Agent: The complete autonomous workflow with self-correction (DEBUG MODE).
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState
from agent.tools import execute_sql_query, retrieve_policy_context
from agent.router import route_question
from agent.generator import generate_answer
from models import get_llm

# --- HELPER: Clean LLM Output ---
def clean_llm_output(text: str) -> str:
    """Cleans markdown and conversational text from LLM outputs."""
    text = text.replace("```sql", "").replace("```", "")
    if "SELECT" in text.upper():
        start_idx = text.upper().find("SELECT")
        text = text[start_idx:]
    if ";" in text:
        text = text[:text.find(";") + 1]
    return text.strip()

# --- NODE FUNCTIONS WITH DEBUG PRINTS ---

def router_node(state: AgentState) -> AgentState:
    """Decides which path to take based on the question."""
    decision = route_question(state)
    print(f"\n🔍 [ROUTER] Decision: '{decision}'")
    return {"router_decision": decision}

def sql_tool_node(state: AgentState) -> AgentState:
    """Executes SQL query and stores the result."""
    # For now, we'll use a simple prompt to generate SQL
    # In production, you'd want more sophisticated SQL generation
    from langchain_core.prompts import ChatPromptTemplate

    sql_prompt = ChatPromptTemplate.from_template(
        """You are an expert SQL developer. Generate ONLY a valid SQLite SELECT query for this question.
        
        Tables:
        - customers(account_id, customer_age, customer_occupation)
        - accounts(account_id, current_balance, primary_channel)
        - transactions(transaction_id, account_id, transaction_amount, transaction_date, transaction_type, location, device_id, ip_address, merchant_id, login_attempts)
        
        Question: {question}
        
        Return ONLY the SQL query. No explanations. No markdown.
        """
    )

    llm = get_llm()
    response = llm.invoke(sql_prompt.format(question=state["messages"][-1].content))

    sql_query = clean_llm_output(response.content)
    print(f"\n🔍 [SQL TOOL] Generated Query:\n{sql_query}")

    # Execute the query
    result = execute_sql_query.invoke({"sql_query": sql_query})
    print(f"🔍 [SQL TOOL] Result:\n{result[:300]}...") # Print first 300 chars

    return {
        "sql_query": sql_query,
        "sql_result": result
    }

def rag_tool_node(state: AgentState) -> AgentState:
    """Retrieves relevant policy context."""
    query = state["messages"][-1].content
    context = retrieve_policy_context.invoke({"query": query})
    print(f"\n🔍 [RAG TOOL] Retrieved Context (first 300 chars):\n{context[:300]}...")

    return {"rag_context": context}

def both_tools_node(state: AgentState) -> AgentState:
    """Executes both SQL and RAG tools."""
    print("\n🔍 [BOTH TOOLS] Executing SQL and RAG...")

    # Run SQL
    sql_state = sql_tool_node(state)

    # Run RAG
    rag_state = rag_tool_node(state)

    # Combine results
    return {
        **sql_state,
        **rag_state
    }

def generator_node(state: AgentState) -> AgentState:
    """Generates the final answer."""
    print("\n🔍 [GENERATOR] Synthesizing final answer...")
    result = generate_answer(state)
    print(f"🔍 [GENERATOR] Raw LLM Output:\n{result['final_answer'][:400]}...")
    return result

def format_final_response(state: AgentState) -> AgentState:
    """Formats the final answer as a message."""
    # STRICT EXTRACTION: Ensure we get the generated answer, not the question
    final_answer = state.get("final_answer")
    if not final_answer or len(final_answer) < 10:
        final_answer = "I apologize, but I could not generate a complete answer based on the available data."
    
    print(f"\n🔍 [FORMAT] Final Answer extracted: '{final_answer[:100]}...'")
    return {"messages": [AIMessage(content=final_answer)]}

# --- CONDITIONAL EDGES ---

def route_to_tools(state:AgentState) -> Literal["sql_tool", "rag_tool", "both_tools"]:
    """Routes to the appropriate tool based on router decision."""
    decision = state.get("router_decision", "rag")

    if decision == "sql":
        return "sql_tool"
    elif decision == "rag":
        return "rag_tool"
    else: 
        return "both_tools"

# --- BUILD THE GRAPH ---

def build_agent_graph():
    """Constructs the complete LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("sql_tool", sql_tool_node)
    workflow.add_node("rag_tool", rag_tool_node)
    workflow.add_node("both_tools", both_tools_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("format_response", format_final_response)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_conditional_edges(
        "router", 
        route_to_tools,
        {
            "sql_tool": "sql_tool",
            "rag_tool": "rag_tool",
            "both_tools": "both_tools"
        }
    )

    # All tools lead to generator
    workflow.add_edge("sql_tool", "generator")
    workflow.add_edge("rag_tool", "generator")
    workflow.add_edge("both_tools", "generator")

    # Generator leads to formatting
    workflow.add_edge("generator", "format_response")

    # Format response leads to END
    workflow.add_edge("format_response", END)

    # Compile the graph
    return workflow.compile()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("Building Finance Agent Graph...")
    agent = build_agent_graph()
    print("Agent graph built successfully!\n")

    # Test the agent
    test_questions = [
        "How many transactions over $5000 happened in the dataset?",
        "What are the Red Flag Indicators for fraud?",
        "Calculate the total transaction amount for account 10001 and tell me the policy for account verification."
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {question}")
        print('='*60)

        result = agent.invoke({
            "messages": [HumanMessage(content=question)],
            "retry_count": 0
        })

        final_msg = result['messages'][-1]
        print(f"\n✅ FINAL ANSWER:\n{final_msg.content}")
        print('='*60)