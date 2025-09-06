# Global in-memory session store
import pdb
SESSION_STORE = {}


# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.callbacks.tracers.langchain import LangChainTracer
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


from agent.agentic_logic import (
    llm,
    agent_prompt,
    agent_runnable,
    build_initial_state,
    run_agent,
    handle_general_query,
    handle_hiring_query,
    profile_match,
    collect_job_details,
    generate_job_description,
    get_hiring_plan_role_and_purpose,
    get_hiring_plan_time_and_urgency,
    get_hiring_plan_work_authorization_requirements,
    generate_hiring_checklist,
    AgentState
)

# ─── Load environment ──────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)
tracer = LangChainTracer()

os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "LangGraph_HRAgent"


# ─── LangGraph Workflow ────────────────────────────────────────────────────────
checkpointer = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("initialize_agent", run_agent)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("profile_match", profile_match)
workflow.add_node("handle_hiring_query", handle_hiring_query)
workflow.add_node("collect_job_details", collect_job_details) 
workflow.add_node("get_hiring_plan_role_and_purpose", get_hiring_plan_role_and_purpose)
workflow.add_node("get_hiring_plan_time_and_urgency", get_hiring_plan_time_and_urgency)
workflow.add_node("get_hiring_plan_work_authorization_requirements", get_hiring_plan_work_authorization_requirements)
workflow.add_node("generate_job_description", generate_job_description)
workflow.add_node("generate_hiring_checklist", generate_hiring_checklist)

workflow.add_conditional_edges(
    "initialize_agent",
    lambda s: (
        "handle_hiring_query" if s["intent"] == "hiring"
        else "handle_general_query" if s["intent"] == "general_query"
        else "end"
    )
)


# Normal internal flow after resume

workflow.add_conditional_edges(
    "handle_hiring_query",
    lambda s: (
        "collect_job_details"
        if s.get("hiring_support_option") == 1 else
        "profile_match"
        if s.get("hiring_support_option") == 2 else
        "get_hiring_plan_role_and_purpose"
    )
)

workflow.add_conditional_edges(
    "get_hiring_plan_role_and_purpose",
    lambda s: (
        "get_hiring_plan_role_and_purpose"
        if not s.get("hiring_plan_details_role_and_purpose_complete") else
        "get_hiring_plan_time_and_urgency"
    )
)

workflow.add_conditional_edges(
    "get_hiring_plan_time_and_urgency",
    lambda s: (
        "get_hiring_plan_time_and_urgency"
        if not s.get("hiring_plan_details_time_and_urgency_complete") else
        "get_hiring_plan_work_authorization_requirements"
    )
)

workflow.add_conditional_edges(
    "get_hiring_plan_work_authorization_requirements",
    lambda s: (
        "get_hiring_plan_work_authorization_requirements"
        if not s.get("hiring_plan_details_work_authorization_complete") else    
        "generate_hiring_checklist"
    )
)


workflow.add_conditional_edges(
    "generate_hiring_checklist",
    lambda s: "end",
    {"end": END}
)

workflow.add_conditional_edges(
    "collect_job_details",
    lambda s: "collect_job_details" if s.get("job_details_missing") else "generate_job_description"
)


workflow.add_conditional_edges(
    "generate_job_description",
    # if additional_drafts is True, loop back; otherwise finish
    lambda s: "generate_job_description" if s.get("additional_drafts") else "end",
    {
      "generate_job_description": "generate_job_description",
      "end": END
    }
)

workflow.add_conditional_edges(
    "handle_general_query",
    lambda s: "end",
    {"end": END}
)

workflow.set_entry_point("initialize_agent")
graph = workflow.compile(
    checkpointer=checkpointer, #Needed for resuming sessions
)

# ─── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
# Allow requests from your Next.js dev server (localhost:3000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.post("/api/find-reaction")
def find_reaction():
    data = request.get_json(silent=True) or {}
    persona_id = data.get("persona_id", 3)
    text = data.get("text", "")

    # Static demo response (you can swap this with real logic later)
    resp = {
        "persona_id": persona_id,
        "similarity": 0.433674156665802,
        "matched_row_id": 4170,
        "used_attributes": True,
        "top_emotion": "neutral",
        "emotion_scores": [
            {"label": "neutral", "confidence": 0.915312171}
        ],
        "masking": 1,
        "attributes_sent": {
            "id": 4170,
            "top_emotion": "neutral",
            "reddit_sentiment": "positive",
            "emo1_label": "neutral",
            "emo1_conf": 0.915312171,
            "horoscope": "Aquarius",
            "job": "Military",
            "masking": 1,
            "cluster": 3,
            "x2d": -0.6668546483,
            "y2d": 0.9311182029,
            "cluster_color_name": "lightsalmon",
            "persona_id": persona_id
        },
        "high_conf_emotions_sent": [
            {"label": "neutral", "confidence": 0.915312171}
        ],
        "reaction": (
            "It’s a complex situation; while the father's actions may stem from a desire to "
            "protect his child, the consequences of taking a life are severe and troubling. "
            "The approval of his wife's perspective adds another layer to the emotional weight of this case."
        )
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

