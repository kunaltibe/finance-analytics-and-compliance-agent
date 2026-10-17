"""
Generator Node: Synthesizes tool outputs into a final answer.
"""
from langchain_core.prompts import ChatPromptTemplate
from models import get_llm

def generate_answer(state: dict) -> dict:
    """
    Takes the intermediate results (SQL, RAG) and generates a final answer.
    """
    generator_prompt = ChatPromptTemplate.from_template(
        """You are a helpful financial services assistant. Answer the user's question based STRICTLY on the provided context.

User Question: {question}

SQL Query Results (if available):
{sql_result}

Policy Context (if available):
{rag_context}

Instructions:
1. EXPLICITLY restate the user's criteria in your opening sentence (e.g., "Based on the data, the accounts with >200 login attempts and >$1000 transactions are...").
2. If the SQL result is empty or lacks specific policy thresholds, explicitly state: "The database query returned no matches for this specific criteria" or "The provided policy documents do not specify a threshold for this." DO NOT guess or hallucinate rules.
3. Combine SQL data and RAG policy data seamlessly if both are present.
4. Use clean markdown formatting.

Provide your answer:
"""
    )

    llm = get_llm(temperature = 0.0)
    chain = generator_prompt | llm

    response = chain.invoke({
        "question": state["messages"][-1].content,
        "sql_result": state.get("sql_result", "No SQL data available"),
        "rag_context": state.get("rag_context", "No policy context available")
    })

    return {"final_answer": response.content}