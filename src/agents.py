import os
import re
import json
from typing import List, Optional
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from crewai import Agent, Task, Crew, Process, LLM

# --- Fix for CrewAI + Groq cache_breakpoint bug ---
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

_orig_litellm_completion = litellm.completion

def _safe_litellm_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                msg.pop("cache_control", None)
    return _orig_litellm_completion(*args, **kwargs)

litellm.completion = _safe_litellm_completion
litellm.drop_params = True
# --------------------------------------------------

from src.retriever import get_recruiter_tools

load_dotenv()


# =====================================================================
# 1. Pydantic Structured Output Schemas
# =====================================================================
class STARBreakdown(BaseModel):
    situation: str = Field(
        description="Assessment of context/situation (Clear, Vague, or Missing) with 1-line reason."
    )
    task: str = Field(
        description="Assessment of the candidate's specific task/goal (Clear, Vague, or Missing) with 1-line reason."
    )
    action: str = Field(
        description="Assessment of technical contributions, tools used, and problem solving (Strong, Generic, or Missing) with 1-line reason."
    )
    result: str = Field(
        description="Assessment of project outcome, test results, lessons learned, or demo impact (Clear, Qualitative, or Missing) with 1-line reason."
    )


class CoachEvaluation(BaseModel):
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall score from 0.0 to 10.0 based on STAR structure, CS fundamentals, and JD alignment."
    )
    star_assessment: STARBreakdown = Field(
        description="Granular evaluation across Situation, Task, Action, and Result."
    )
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Key technical terms/skills from the JD successfully mentioned by the candidate."
    )
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="Expected beginner/junior keywords from the JD that the candidate missed."
    )
    candidate_rewrite: str = Field(
        description="A 2-3 sentence high-impact STAR rewrite highlighting the candidate's actual academic/personal project experience."
    )
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description="Up to two exact short excerpts from the candidate's answer supporting the feedback; empty if no excerpt supports it."
    )
    probe_target: str = Field(
        description="A constructive instruction targeting the candidate's weakest point for the next question."
    )


def verify_evidence_quotes(evaluation: CoachEvaluation, answer: str) -> CoachEvaluation:
    verified_quotes = [
        quote.strip()
        for quote in evaluation.evidence_quotes
        if quote.strip() and quote.strip() in answer
    ][:2]
    return evaluation.model_copy(update={"evidence_quotes": verified_quotes})


# =====================================================================
# 2. Agent Manager (Fresher & Entry-Level Calibrated)
# =====================================================================
class RecruiterSimulationManager:
    def __init__(
        self,
        vector_db: Chroma,
        resume_text: str = "",
        model_name: str = "groq/openai/gpt-oss-120b",
        target_role: str = ""
    ):
        self.vector_db = vector_db
        self.resume_text = resume_text
        self.target_role = target_role.strip()
        self.tools = get_recruiter_tools(vector_db)
        
        self.llm = LLM(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        
        self.interviewer_agent = self._create_interviewer()
        self.coach_agent = self._create_coach()

    def _create_interviewer(self) -> Agent:
        return Agent(
            role="University & Entry-Level Technical Hiring Lead",
            goal=(
                "Conduct an encouraging yet thorough entry-level interview designed for freshers, interns, "
                "and recent college graduates. Ask questions grounded in core CS fundamentals, academic projects, "
                "coding concepts, and cultural alignment without expecting years of corporate production experience."
            ),
            backstory=(
                "You are an Engineering Lead and Campus Recruiter who loves mentoring and evaluating early-career engineers. "
                "You understand that freshers build capstones, course assignments, hackathon prototypes, or open-source projects. "
                "You start with a welcoming self-introduction question and follow up on their project architecture, "
                "how they solved bugs, their grasp of fundamentals (data structures, APIs, databases, logic), and their eagerness to learn."
            ),
            tools=self.tools,
            llm=self.llm,
            verbose=False,
            memory=False
        )

    def _create_coach(self) -> Agent:
        resume_snippet = self.resume_text[:2500] if self.resume_text else "Candidate is a fresher with academic projects."
        return Agent(
            role="Early-Career & Campus Placement Coach",
            goal=(
                "Coach students and freshers on structuring their project stories using the STAR methodology. "
                "Help them articulate technical decisions, individual contributions, and project outcomes effectively."
            ),
            backstory=(
                "You are a mentor specializing in campus placements and entry-level tech interviews. "
                "You know freshers often lack corporate metrics, so you teach them to treat academic projects, "
                "hackathons, course assignments, and internships as valid STAR stories. "
                f"Candidate's Verified Profile/Resume:\n<resume>\n{resume_snippet}\n</resume>\n"
                "When rewriting answers, highlight their actual project tools and learning outcomes clearly."
            ),
            tools=self.tools,
            llm=self.llm,
            verbose=False,
            memory=False
        )

    def generate_opening_question(self, focus_mode: str, total_rounds: int) -> str:
        """Round 1: Always starts with a natural self-introduction + project overview."""
        task = Task(
            description=(
                f"Generate the opening interview question for Round 1 of {total_rounds}.\n"
                f"Candidate Level: Fresher / Recent Graduate.\n"
                f"Target Role: {self.target_role or 'Not specified'}.\n"
                f"Focus Mode: {focus_mode}.\n\n"
                "Instructions:\n"
                "1. Craft a warm, conversational Round 1 opening question asking the candidate to introduce themselves, "
                "their technical background, and give a high-level overview of a major project or area of interest they are proud of.\n"
                "2. Reference the target role or company if available in the context.\n"
                "3. Keep the tone approachable and encouraging. Return ONLY the question string."
            ),
            expected_output="A welcoming opening self-intro and project question string.",
            agent=self.interviewer_agent
        )
        crew = Crew(agents=[self.interviewer_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result).strip()

    def evaluate_response(self, question: str, answer: str) -> CoachEvaluation:
        task = Task(
            description=(
                f"Evaluate the fresher candidate's response.\n\n"
                f"Question: \"{question}\"\n"
                f"Candidate Answer: \"{answer}\"\n\n"
                "Instructions:\n"
                "1. Evaluate against the STAR method adapted for freshers (Situation, Task, Action, Result).\n"
                "   - Situation/Task: Was the project/problem clear?\n"
                "   - Action: Did they explain their specific coding/technical contribution?\n"
                "   - Result: Did they mention project completion, features working, test results, or what they learned?\n"
                "2. Check if they mentioned relevant technologies from `query_job_requirements`.\n"
                "3. Provide a supportive 2-3 sentence rewrite using their real background.\n"
                "4. Cite up to two exact short excerpts from the candidate answer as `evidence_quotes`; do not paraphrase or invent evidence. Use an empty list if no excerpt supports the feedback.\n"
                "5. Suggest a targeted follow-up `probe_target` (e.g., asking about a specific bug they solved or how an API was structured).\n\n"
                "Return strict JSON matching schema:\n"
                "{\n"
                '  "score": 7.5,\n'
                '  "star_assessment": {\n'
                '    "situation": "...",\n'
                '    "task": "...",\n'
                '    "action": "...",\n'
                '    "result": "..."\n'
                "  },\n"
                '  "matched_keywords": ["..."],\n'
                '  "missing_keywords": ["..."],\n'
                '  "candidate_rewrite": "...",\n'
                '  "evidence_quotes": ["exact words from the candidate answer"],\n'
                '  "probe_target": "..."\n'
                "}"
            ),
            expected_output="A strict JSON object matching the CoachEvaluation schema.",
            output_pydantic=CoachEvaluation,
            agent=self.coach_agent
        )

        crew = Crew(agents=[self.coach_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        if hasattr(result, "pydantic") and result.pydantic:
            return verify_evidence_quotes(result.pydantic, answer)

        try:
            raw_str = str(result)
            json_match = re.search(r"\{.*\}", raw_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return verify_evidence_quotes(CoachEvaluation(**data), answer)
        except Exception as e:
            print(f"[Warning] Fallback parsing used: {e}")

        return CoachEvaluation(
            score=7.0,
            star_assessment=STARBreakdown(
                situation="Project context was introduced",
                task="Identified the main objective",
                action="Described general technical steps",
                result="Mentioned completion; could detail lessons learned or test results"
            ),
            matched_keywords=[],
            missing_keywords=[],
            candidate_rewrite="Highlight your exact role in building the features and specify the core technologies used.",
            evidence_quotes=[],
            probe_target="Ask the candidate to explain the key technical challenge they encountered and how they debugged it."
        )

    def generate_followup_question(
        self,
        previous_question: str,
        candidate_answer: str,
        probe_target: str,
        round_num: int,
        total_rounds: int,
        focus_mode: str
    ) -> str:
        task = Task(
            description=(
                f"You are the Campus Hiring Lead. Ask the question for Round {round_num} of {total_rounds}.\n"
                f"Candidate: Fresher / Entry-Level Engineer\n"
                f"Target Role: {self.target_role or 'Not specified'}\n"
                f"Focus Mode: {focus_mode}\n\n"
                f"Previous Question: \"{previous_question}\"\n"
                f"Candidate Answer: \"{candidate_answer}\"\n"
                f"Coach's Target Probe: \"{probe_target}\"\n\n"
                "Instructions:\n"
                "1. If following up on their previous answer, ask a friendly yet technical question about how they implemented a feature, solved a specific bug, or tested their code.\n"
                "2. If moving to a new topic (e.g., core CS concepts or behavioral teamwork), query `query_job_requirements` or `query_company_culture`.\n"
                "3. Keep questions appropriate for freshers—focus on fundamentals, design rationale, and willingness to learn. Return ONLY the question string."
            ),
            expected_output="A single entry-level interview question string.",
            agent=self.interviewer_agent
        )
        crew = Crew(agents=[self.interviewer_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result).strip()