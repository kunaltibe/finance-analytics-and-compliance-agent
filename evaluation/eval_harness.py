 # evaluation/eval_harness.py
"""
Automated Evaluation Harness for the Finance Agent.
Scores agent performance on Routing Accuracy, Faithfulness, and Correctness.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from models import get_llm
from agent.graph import build_agent_graph
from langchain_core.messages import HumanMessage

# --- 1. THE GOLDEN DATASET ---
# In production, this would be a JSON file. Here, we define it inline for the demo.

GOLDEN_DATASET = [
    {
        "id": 1,
        "question": "Show me all accounts that had more than 200 login attempts and a transaction amount over 1000.",
        "expected_route": "sql",
        "expected_answer_contains": ["account", "login", "1000"]
    },
    {
        "id": 2,
        "question": "According to the 2026 Anti-Fraud Policy, what is the immediate containment SOP for accounts with Red Flag Indicators?",
        "expected_route": "rag",
        "expected_answer_contains": ["containment", "SOP", "verification", "identification"]
    },
    {
        "id": 3,
        "question": "Find the top 3 accounts with the highest transaction amounts, and tell me if the fraud policy requires legal audits for high-value accounts.",
        "expected_route": "both",
        "expected_answer_contains": ["account", "legal audit", "high value", "policy"]
    }
]

# --- 2. THE LLM-AS-A-JUDGE PROMPTS ---

def evaluate_faithfulness(question: str, context: str, answer: str) -> dict:
    """Checks if the answer is grounded in the provided context (SQL/RAG) without hallucination."""
    prompt = ChatPromptTemplate.from_template(
        """You are an expert AI evaluator. Determine if the 'Answer' is strictly grounded in the 'Context' provided.
        
        Question: {question}
        Context: {context}
        Answer: {answer}
        
        Score the Faithfulness from 1 to 5:
        1 = Complete hallucination, ignores context.
        3 = Partially grounded, but adds some unsupported claims.
        5 = Perfectly grounded, every claim is supported by the context.
        
        Respond ONLY with a JSON object: {{"score": <int>, "reasoning": "<short explanation>"}}
        """
    )
    llm = get_llm(temperature=0.0) # Deterministic for grading
    response = llm.invoke(prompt.format(question=question, context=context, answer=answer))

    try:
        # Clean up markdown if the LLM adds it
        clean_text = response.content.replace("```json", "").replace("```", "")
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": "Failed to parse LLM JSON response"}

def evaluate_correctness(question: str, answer: str, expected_keywords: list) -> dict:
    """Checks if the answer actually addresses the user's question and contains expected concepts."""
    prompt = ChatPromptTemplate.from_template(
        """You are an expert AI evaluator. Determine if the 'Answer' correctly and comprehensively addresses the 'Question'.
        
        Question: {question}
        Answer: {answer}
        Expected Concepts to look for (semantically): {expected_keywords}
        
        Score the Correctness from 1 to 5:
        1 = Does not answer the question or is completely wrong.
        3 = Answers the question but misses some key expected concepts.
        5 = Comprehensively answers the question and includes the expected concepts (semantic match is fine, exact keyword match is not strictly required).
        
        Respond ONLY with a JSON object: {{"score": <int>, "reasoning": "<short explanation>"}}
        """
    )
    llm = get_llm(temperature=0.0)
    response = llm.invoke(prompt.format(question=question, answer=answer, expected_keywords=expected_keywords))

    try:
        clean_text = response.content.replace("```json", "").replace("```", "")
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": "Failed to parse LLM JSON response"}

# --- 3. THE EVALUATION RUNNER ---

def run_evaluation():
    print("=" * 70)
    print(" 🚀 STARTING AUTOMATED AGENT EVALUATION")
    print("=" * 70)

    agent = build_agent_graph()
    results = []
    total_faithfulness = 0
    total_correctness = 0

    for item in GOLDEN_DATASET:
        print(f"\n[Evaluating ID: {item['id']}]")
        print(f"Question: {item['question']}")

        # 1. Invoke the agent
        result = agent.invoke({
            "messages": [HumanMessage(content=item['question'])],
            "retry_count": 0
        })

        final_answer = result['messages'][-1].content
        router_decision = result.get('router_decision', 'unknown')
        sql_result = result.get('sql_result', 'No SQL executed')
        rag_context = result.get('rag_context', 'No RAG executed')

        # Combine context for faithfulness check
        combined_context = f"SQL Result: {sql_result}\n\nRAG Context: {rag_context}"

        # 2. Check Routing Accuracy
        route_score = 1 if router_decision == item["expected_route"] else 0
        print(f"  ➔ Routing: Expected '{item['expected_route']}', Got '{router_decision}' ({'✅' if route_score else '❌'})")

        # 3. Evaluate Faithfulness
        faithfulness = evaluate_faithfulness(item["question"], combined_context, final_answer)
        total_faithfulness += faithfulness["score"]
        print(f"  ➔ Faithfulness: {faithfulness['score']}/5 - {faithfulness['reasoning']}")

        # 4. Evaluate Correctness
        correctness = evaluate_correctness(item["question"], final_answer, item["expected_answer_contains"])
        total_correctness += correctness["score"]
        print(f"  ➔ Correctness: {correctness['score']}/5 - {correctness['reasoning']}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "router_decision": router_decision,
            "faithfulness_score": faithfulness["score"],
            "correctness_score": correctness["score"],
            "final_answer": final_answer[:100] + "..."      # Truncate for report
        })

    # --- 4. FINAL REPORT ---
    num_tests = len(GOLDEN_DATASET)
    print("\n" + "=" * 70)
    print(" 📊 EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Tests Run: {num_tests}")
    print(f"Routing Accuracy: {sum(1 for r in results if r['router_decision'] == GOLDEN_DATASET[r['id']-1]['expected_route'])}/{num_tests}")
    print(f"Average Faithfulness Score: {total_faithfulness/num_tests:.2f}/5.0")
    print(f"Average Correctness Score: {total_correctness/num_tests:.2f}/5.0")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_evaluation()