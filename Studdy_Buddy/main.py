import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv() 

llm = ChatGroq(model="qwen/qwen3-32b")


# ---- Shared state -----------------------------------------------------
class StudyState(TypedDict):
    topic: str
    research: str
    quiz: str
    answers: str
    feedback: str
    simplified: str


# ---- Agent 1: Researcher ----------------------------------------------
def researcher(state: StudyState) -> dict:
    print("\n[Researcher] Looking up key points...")
    prompt = (
        f"You are a research agent helping a student study '{state['topic']}'. "
        "Write a concise, well-organized summary covering 5-8 key points. "
        "Use short paragraphs or a numbered list. Keep it factual and clear."
    )
    response = llm.invoke(prompt)
    return {"research": response.content}


# ---- Agent 2: Quizzer ---------------------------------------------------
def quizzer(state: StudyState) -> dict:
    print("[Quizzer] Writing quiz questions...")
    prompt = (
        "You are a quiz-writing agent. Based ONLY on the research notes below, "
        "write exactly 3 quiz questions to test understanding of the topic. "
        "Number them 1-3. Do NOT include the answers.\n\n"
        f"Research notes:\n{state['research']}"
    )
    response = llm.invoke(prompt)
    return {"quiz": response.content}


# ---- Collect the user's answers (not an LLM agent, just I/O) ----------
def collect_answers(state: StudyState) -> dict:
    print("\n--- Quiz ---")
    print(state["quiz"])
    print("\nType your answer to each question. When finished, type DONE on its own line.")

    lines = []
    while True:
        line = input("> ")
        if line.strip().upper() == "DONE":
            break
        lines.append(line)

    return {"answers": "\n".join(lines)}


# ---- Agent 3: Grader -----------------------------------------------------
def grader(state: StudyState) -> dict:
    print("\n[Grader] Checking your answers...")
    prompt = (
        "You are a grading agent. Using the research notes as the source of "
        "truth, grade the student's answers to the quiz questions. For each "
        "question, say whether the answer was correct, partially correct, or "
        "incorrect, with a one-line explanation. Then give an overall score.\n\n"
        f"Research notes:\n{state['research']}\n\n"
        f"Quiz questions:\n{state['quiz']}\n\n"
        f"Student's answers:\n{state['answers']}\n\n"
        "End your response with a line in EXACTLY this format:\n"
        "SCORE: X/10"
    )
    response = llm.invoke(prompt)
    return {"feedback": response.content}


# ---- Bonus Agent: Simplifier --------------------------------------------
def simplifier(state: StudyState) -> dict:
    print("[Simplifier] Score was low -- re-explaining in simpler terms...")
    prompt = (
        f"A student is struggling with the topic '{state['topic']}'. Here is "
        f"the feedback on their quiz:\n\n{state['feedback']}\n\n"
        "Re-explain the topic in much simpler terms. Use everyday analogies "
        "and focus especially on the areas where the student struggled."
    )
    response = llm.invoke(prompt)
    return {"simplified": response.content}


# ---- Routing: decide whether the Simplifier runs ------------------------
def route_after_grading(state: StudyState) -> str:
    match = re.search(r"SCORE:\s*(\d+)", state["feedback"])
    score = int(match.group(1)) if match else 10
    return "simplifier" if score < 6 else "end"


# ---- Build the graph -----------------------------------------------------
def build_graph():
    graph = StateGraph(StudyState)

    graph.add_node("researcher", researcher)
    graph.add_node("quizzer", quizzer)
    graph.add_node("collect_answers", collect_answers)
    graph.add_node("grader", grader)
    graph.add_node("simplifier", simplifier)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "quizzer")
    graph.add_edge("quizzer", "collect_answers")
    graph.add_edge("collect_answers", "grader")
    graph.add_conditional_edges(
        "grader",
        route_after_grading,
        {"simplifier": "simplifier", "end": END},
    )
    graph.add_edge("simplifier", END)

    return graph.compile()


if __name__ == "__main__":
    topic = input("What topic do you want to study today? ").strip()

    app = build_graph()
    result = app.invoke(
        {
            "topic": topic,
            "research": "",
            "quiz": "",
            "answers": "",
            "feedback": "",
            "simplified": "",
        }
    )

    print("\n--- Research notes ---")
    print(result["research"])

    print("\n--- Feedback ---")
    print(result["feedback"])

    if result.get("simplified"):
        print("\n--- Simplified explanation ---")
        print(result["simplified"])
