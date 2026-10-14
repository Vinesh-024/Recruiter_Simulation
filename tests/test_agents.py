import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents import RecruiterSimulationManager

load_dotenv()

def test_agent_loop():
    print("Initializing Groq Agents Test...")
    
    # Connect to the local Chroma collection
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="./chromadb_store/test_run_01",
        embedding_function=embeddings,
        collection_name="recruiter_test_run_01"
    )

    manager = RecruiterSimulationManager(
        vector_db=db,
        resume_text="Built a real-time analytics backend using Python, Redis, and FastAPI handling 5k req/sec."
    )

    # 1. Test Opening Question
    print("\n--- Generating Opening Question (Round 1) ---")
    question = manager.generate_opening_question(
        focus_mode="Mixed (Behavioral + Hard Skills)",
        total_rounds=3
    )
    print(f"Interviewer Question:\n{question}\n")

    # 2. Test Candidate Answer Evaluation
    sample_answer = "We experienced API slowdowns under peak load, so I introduced Redis caching to reduce database read latency."
    print("--- Evaluating Sample Candidate Answer ---")
    evaluation = manager.evaluate_response(question=question, answer=sample_answer)
    
    print(f"Score: {evaluation.score} / 10")
    print(f"STAR Assessment: {evaluation.star_assessment.model_dump()}")
    print(f"Matched Keywords: {evaluation.matched_keywords}")
    print(f"Missing Keywords: {evaluation.missing_keywords}")
    print(f"Coach Rewrite:\n{evaluation.candidate_rewrite}\n")
    print(f"Probe Target: {evaluation.probe_target}\n")

    # 3. Test Adaptive Follow-up
    print("--- Generating Adaptive Follow-up Question (Round 2) ---")
    followup = manager.generate_followup_question(
        previous_question=question,
        candidate_answer=sample_answer,
        probe_target=evaluation.probe_target,
        round_num=2,
        total_rounds=3,
        focus_mode="Mixed (Behavioral + Hard Skills)"
    )
    print(f"Interviewer Follow-up:\n{followup}")

if __name__ == "__main__":
    test_agent_loop()