import sys
import os

# Ensure project root is in Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import uuid
from langchain_groq import ChatGroq
from pypdf import PdfReader
import streamlit as st

from src.agents import CoachEvaluation, RecruiterSimulationManager
from src.multi_loader import MultiSourceIngestionEngine

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fresher Recruiter Simulator | Entry-Level AI Mock Interview",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. State Initialization
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "current_round" not in st.session_state:
    st.session_state.current_round = 1
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "manager" not in st.session_state:
    st.session_state.manager = None
if "ingest_metadata" not in st.session_state:
    st.session_state.ingest_metadata = {}
if "round_questions" not in st.session_state:
    st.session_state.round_questions = []
if "practice_retries" not in st.session_state:
    st.session_state.practice_retries = []
if "retry_notice" not in st.session_state:
    st.session_state.retry_notice = ""

# -----------------------------------------------------------------------------
# 3. Sidebar: Configuration & Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Fresher Mock Interview")

    st.subheader("1. Target Role & Company")
    target_role = st.text_input(
        "Target role",
        placeholder="e.g. Junior Python Developer"
    )
    jd_text_input = st.text_area(
        "Paste job description",
        placeholder="Paste the role description here for the most reliable role-specific practice.",
        height=120
    )
    culture_urls_input = st.text_area(
        "Company About / Culture URLs",
        placeholder="https://company.com/about\nhttps://company.com/culture",
        height=70,
        help="Links to the company's values, mission, or culture pages."
    )
    jd_urls_input = st.text_area(
        "Job Description / Career URLs",
        placeholder="https://boards.greenhouse.io/...\nhttps://jobs.lever.co/...",
        height=70,
        help="Direct links to entry-level / graduate job postings."
    )
    uploaded_jd_pdfs = st.file_uploader(
        "Upload Campus Placement / JD PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload campus placement brochures or junior job descriptions."
    )

    st.subheader("2. Candidate Profile (Fresher / Student)")
    uploaded_resume = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"],
        help="Upload your resume with academic projects, internships, hackathons, or coursework."
    )
    if uploaded_resume:
        try:
            reader = PdfReader(uploaded_resume)
            st.session_state.resume_text = "\n".join(
                [p.extract_text() for p in reader.pages if p.extract_text()]
            )
            st.caption(f"✓ Resume loaded ({len(st.session_state.resume_text.split())} words)")
        except Exception as e:
            st.error(f"Error reading resume: {e}")

    st.subheader("3. Session Settings")
    focus_mode = st.selectbox(
        "Interview Focus",
        [
            "Mixed (Self-Intro + Tech Projects + Core CS)",
            "Project & Technical Fundamentals",
            "Behavioral & Culture Fit"
        ]
    )
    max_rounds = st.slider("Total Rounds", min_value=3, max_value=6, value=4)

    # Ingestion Trigger
    if st.button("🚀 Start Interview", type="primary", use_container_width=True):
        jd_urls = [u.strip() for u in jd_urls_input.split("\n") if u.strip()]
        culture_urls = [u.strip() for u in culture_urls_input.split("\n") if u.strip()]

        if not (jd_urls or culture_urls or uploaded_jd_pdfs or jd_text_input.strip()):
            st.error("Paste a job description, add a job or culture URL, or upload a JD PDF to begin.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found in .env.")
        else:
            with st.spinner("Preparing interview session grounded in job requirements..."):
                try:
                    st.session_state.session_id = uuid.uuid4().hex
                    engine = MultiSourceIngestionEngine(session_id=st.session_state.session_id)
                    ingest_res = engine.ingest_all(
                        jd_urls=jd_urls,
                        culture_urls=culture_urls,
                        uploaded_pdfs=uploaded_jd_pdfs,
                        raw_text_snippets=(
                            [{
                                "title": target_role.strip() or "Pasted job description",
                                "text": jd_text_input.strip(),
                                "doc_type": "job_description",
                            }]
                            if jd_text_input.strip()
                            else None
                        )
                    )

                    vector_db = ingest_res["vector_db"]
                    st.session_state.ingest_metadata = {
                        "chunks": ingest_res["total_chunks"],
                        "docs": ingest_res["total_documents"],
                        "sources": ingest_res["sources"]
                    }

                    # Instantiate Orchestration Manager
                    manager = RecruiterSimulationManager(
                        vector_db=vector_db,
                        resume_text=st.session_state.resume_text,
                        target_role=target_role
                    )
                    st.session_state.manager = manager

                    # Generate opening question (Self intro & project overview)
                    first_question = manager.generate_opening_question(
                        focus_mode=focus_mode,
                        total_rounds=max_rounds
                    )

                    # Reset Session State
                    st.session_state.current_question = first_question
                    st.session_state.round_questions = [first_question]
                    st.session_state.chat_history = [{"role": "interviewer", "content": first_question}]
                    st.session_state.evaluations = []
                    st.session_state.practice_retries = []
                    st.session_state.retry_notice = ""
                    st.session_state.current_round = 1
                    st.session_state.turn_counter = 0
                    st.session_state.interview_active = True
                    st.session_state.interview_complete = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization failed: {e}")

    # Telemetry Monitor
    if st.session_state.ingest_metadata:
        with st.expander("📡 Ingestion Telemetry", expanded=False):
            meta = st.session_state.ingest_metadata
            st.write(f"**Indexed Chunks:** {meta.get('chunks', 0)}")
            st.write(f"**Total Sources:** {meta.get('docs', 0)}")
            for src in meta.get("sources", []):
                st.caption(f"• `{src[:35]}...`" if len(src) > 35 else f"• `{src}`")

    if st.button("Delete current practice data", use_container_width=True):
        manager = st.session_state.get("manager")
        if manager is not None:
            manager.vector_db.delete_collection()
        st.session_state.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 4. Main Workspace Layout
# -----------------------------------------------------------------------------
if not st.session_state.interview_active:
    st.markdown("""
    ## 🎓 Fresher & Entry-Level Mock Interview Simulator
    Practice realistic technical and behavioral job interviews tailored specifically for **college students, fresh graduates, and junior engineers**.

    ### Key Highlights
    * **Fresher-Centric Questions:** Starts with a natural self-introduction and asks questions based on your academic projects, hackathons, and CS fundamentals (instead of assuming years of corporate experience).
    * **STAR Method Coaching:** Real-time feedback that teaches you how to present academic projects, coursework, and personal builds using the **STAR** (Situation, Task, Action, Result) framework.
    * **Resume Grounding:** The Coach references your actual uploaded resume to suggest real, practical answer improvements.

    👈 **Add your target job details and upload your resume in the sidebar to begin.**
    """)

elif st.session_state.interview_complete:
    # -------------------------------------------------------------------------
    # Final Scorecard Screen
    # -------------------------------------------------------------------------
    st.header("🎉 Interview Session Complete")
    st.caption("Scores are coaching estimates for practice, not hiring decisions.")

    if st.session_state.retry_notice:
        st.success(st.session_state.retry_notice)
        st.session_state.retry_notice = ""

    avg_score = (
        sum(e.score for e in st.session_state.evaluations) / len(st.session_state.evaluations)
        if st.session_state.evaluations else 0.0
    )

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Overall Average Score", f"{avg_score:.1f} / 10")
    col_metric2.metric("Total Rounds Completed", len(st.session_state.evaluations))
    col_metric3.metric("Session ID", st.session_state.session_id)

    st.subheader("📋 Comprehensive Turn-by-Turn Diagnostic")
    for idx, eval_item in enumerate(st.session_state.evaluations, 1):
        with st.expander(f"Round {idx} Feedback — Score: {eval_item.score}/10", expanded=True):
            if idx <= len(st.session_state.round_questions):
                st.markdown(f"**Question:** {st.session_state.round_questions[idx - 1]}")
            st.markdown("**STAR Framework Assessment:**")
            st.markdown(f"- **Situation:** {eval_item.star_assessment.situation}")
            st.markdown(f"- **Task:** {eval_item.star_assessment.task}")
            st.markdown(f"- **Action:** {eval_item.star_assessment.action}")
            st.markdown(f"- **Result:** {eval_item.star_assessment.result}")

            st.markdown(
                "**Matched Keywords:** " + " ".join([f"`{k}`" for k in eval_item.matched_keywords])
                if eval_item.matched_keywords else "**Matched Keywords:** *None identified*"
            )
            st.markdown(
                "**Missing Opportunities:** " + " ".join([f"`{k}`" for k in eval_item.missing_keywords])
                if eval_item.missing_keywords else "**Missing Opportunities:** *None identified*"
            )

            st.info(f"**Coach's Project-Grounded Rewrite:**\n\n{eval_item.candidate_rewrite}")
            if eval_item.evidence_quotes:
                st.markdown("**Evidence from your answer:**")
                for quote in eval_item.evidence_quotes:
                    st.code(quote)

            with st.form(f"retry_round_{st.session_state.session_id}_{idx}"):
                retry_answer = st.text_area(
                    "Revise your answer",
                    key=f"retry_answer_{st.session_state.session_id}_{idx}",
                    placeholder="Try the answer again using the feedback above."
                )
                retry_submitted = st.form_submit_button("Get feedback on revision")

            if retry_submitted:
                if not retry_answer.strip():
                    st.warning("Write a revised answer before requesting feedback.")
                else:
                    with st.spinner("Reviewing your revised answer..."):
                        revised_evaluation = st.session_state.manager.evaluate_response(
                            question=st.session_state.round_questions[idx - 1],
                            answer=retry_answer.strip()
                        )
                    st.session_state.evaluations[idx - 1] = revised_evaluation
                    st.session_state.practice_retries.append({
                        "round": idx,
                        "answer": retry_answer.strip(),
                    })
                    st.session_state.retry_notice = f"Round {idx} feedback updated from your revised answer."
                    st.rerun()

    transcript_export = {
        "session_id": st.session_state.session_id,
        "average_score": avg_score,
        "dialogue": st.session_state.chat_history,
        "evaluations": [e.model_dump() for e in st.session_state.evaluations],
        "practice_retries": st.session_state.practice_retries,
    }

    st.download_button(
        label="📥 Download Full Transcript & Feedback (JSON)",
        data=json.dumps(transcript_export, indent=2),
        file_name=f"fresher_interview_transcript_{st.session_state.session_id}.json",
        mime="application/json"
    )

    if st.button("🔄 Start New Interview"):
        manager = st.session_state.get("manager")
        if manager is not None:
            manager.vector_db.delete_collection()
        st.session_state.clear()
        st.session_state.interview_active = False
        st.session_state.interview_complete = False
        st.rerun()

else:
    # -------------------------------------------------------------------------
    # Active Two-Column Dashboard
    # -------------------------------------------------------------------------
    st.markdown(
        f"#### 🎯 Round {st.session_state.current_round} of {max_rounds} — Focus: `{focus_mode}`"
    )

    col_chat, col_coach = st.columns([1.1, 0.9], gap="large")

    # Column 1: Live Interview Chat
    with col_chat:
        st.subheader("💬 Live Interview Dialogue")

        chat_box = st.container(height=400)
        with chat_box:
            for message in st.session_state.chat_history:
                if message["role"] == "interviewer":
                    with st.chat_message("assistant", avatar="🎓"):
                        st.markdown(message["content"])
                elif message["role"] == "candidate":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])

        # Dynamic key ensures text area resets clean on every turn
        input_key = f"candidate_ans_r{st.session_state.current_round}_t{st.session_state.turn_counter}"
        user_answer = st.text_area(
            "Your Response:",
            placeholder="Introduce yourself or structure your project answer using STAR (Situation, Task, Action, Result)...",
            height=130,
            key=input_key
        )

        btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1.2, 1])
        with btn_c1:
            submit_answer = st.button("📨 Submit Answer", type="primary", use_container_width=True)
        with btn_c2:
            clarify_request = st.button("❓ Ask Clarification", use_container_width=True)
        with btn_c3:
            end_session = st.button("🛑 End Early", use_container_width=True)

        if end_session:
            st.session_state.interview_complete = True
            st.rerun()

        # Handle Clarification
        if clarify_request:
            with st.spinner("Hiring Lead is clarifying..."):
                groq_chat = ChatGroq(
                    model="openai/gpt-oss-120b",
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
                clarification = groq_chat.invoke(
                    f"The candidate is a fresher asking for clarification on this interview question: "
                    f"\"{st.session_state.current_question}\". "
                    "Provide an encouraging, brief 1-2 sentence clarification or beginner-friendly example without giving away the answer."
                ).content
                st.session_state.chat_history.append(
                    {"role": "interviewer", "content": f"**Clarification:** {clarification}"}
                )
                st.session_state.turn_counter += 1
                st.rerun()

        # Handle Submission
        if submit_answer:
            if not user_answer.strip():
                st.warning("Please enter your answer before submitting.")
            else:
                st.session_state.chat_history.append({"role": "candidate", "content": user_answer})

                with st.spinner("Coach is evaluating your response..."):
                    evaluation = st.session_state.manager.evaluate_response(
                        question=st.session_state.current_question,
                        answer=user_answer
                    )
                    st.session_state.evaluations.append(evaluation)

                if st.session_state.current_round >= max_rounds:
                    st.session_state.interview_complete = True
                else:
                    st.session_state.current_round += 1
                    st.session_state.turn_counter += 1
                    with st.spinner("Hiring Lead is preparing the next question..."):
                        next_q = st.session_state.manager.generate_followup_question(
                            previous_question=st.session_state.current_question,
                            candidate_answer=user_answer,
                            probe_target=evaluation.probe_target,
                            round_num=st.session_state.current_round,
                            total_rounds=max_rounds,
                            focus_mode=focus_mode
                        )
                        st.session_state.current_question = next_q
                        st.session_state.round_questions.append(next_q)
                        st.session_state.chat_history.append({"role": "interviewer", "content": next_q})

                st.rerun()

    # Column 2: Real-time Coach's Feedback Panel
    with col_coach:
        st.subheader("🎯 Real-Time Coaching Feedback")

        if st.session_state.evaluations:
            latest_eval: CoachEvaluation = st.session_state.evaluations[-1]

            st.caption("Feedback is for practice and reflection, not a hiring decision.")
            st.metric("Latest Answer Score", f"{latest_eval.score:.1f} / 10")
            st.progress(min(latest_eval.score / 10.0, 1.0))

            st.markdown("#### ⭐ STAR Framework Compliance")
            star = latest_eval.star_assessment
            st.markdown(f"- **Situation:** {star.situation}")
            st.markdown(f"- **Task:** {star.task}")
            st.markdown(f"- **Action:** {star.action}")
            st.markdown(f"- **Result:** {star.result}")

            st.markdown("#### 🔑 Keyword Coverage")
            if latest_eval.matched_keywords:
                st.markdown("**Matched:** " + " ".join([f"`{k}`" for k in latest_eval.matched_keywords]))
            else:
                st.caption("No matching keywords detected in this answer.")

            if latest_eval.missing_keywords:
                st.markdown("**Suggested Technical Mentions:** " + " ".join([f"`{k}`" for k in latest_eval.missing_keywords]))

            st.markdown("#### 💡 Coach's Recommended Rewrite")
            st.info(latest_eval.candidate_rewrite)

            if latest_eval.evidence_quotes:
                st.markdown("#### Evidence from your answer")
                for quote in latest_eval.evidence_quotes:
                    st.code(quote)

            st.markdown("#### 🔍 Next Focus Area")
            st.caption(f"Interviewer will explore: *{latest_eval.probe_target}*")

        else:
            st.info(
                "👈 Your STAR evaluation, keyword coverage, and personalized project "
                "rewrites will appear here after you submit your first response."
            )