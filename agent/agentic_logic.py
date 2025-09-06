
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langgraph.types import interrupt
from langchain.tools import tool
from typing import Annotated, Union, List, Dict, Any, Optional
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator
import pdb
import json


# ─── Constants ────────────────────────────────────────────────────────────────
from agent.tools import tools, tools_map
from agent.config import COMPANY_DETAILS

# ─── Load environment and instantiate LLM ─────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)


llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))

# ─── Agent prompt and runnable setup ─────────────────────────────────────────
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",  "Answer what ever the user is asking "
    "at the end of the answer mention that though you could do a lot of things"
    "your skills are best used for hiring related purposes"),
    ("user",    "{input}"),
    ("system",  "{agent_scratchpad}")
])


system_query_prompt = ChatPromptTemplate.from_messages([
    ("system",  "Do as asked."),
    ("user",    "{input}"),
    ("system",  "{agent_scratchpad}")
])

HR_ACTIONS = [
    "1. Prepare a job description",
    "2. Get a matching score for profile and job description"
    "3. Create a hiring plan" 
]



agent_runnable = create_openai_functions_agent(llm, tools, agent_prompt)

# ─── Agent State Schema ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    interrupt_count: int
    user_inputs: List[str]
    messages: List[BaseMessage]
    user_messages: List[BaseMessage]
    ################job attributes#################
    job_details: Dict[str, str]
    additional_drafts: bool
    required_fields: List[tuple[str, str]]
    job_details_missing: bool
    generated_job_descriptions: List[str]
    ###############################################
    hiring_support_option: int
    hiring_plan_details: str
    hiring_plan_details_role_and_purpose: str
    hiring_plan_details_time_and_urgency: str
    hiring_plan_details_work_authorization: str
    hiring_plan_details_role_and_purpose_complete: bool
    hiring_plan_details_time_and_urgency_complete: bool
    hiring_plan_details_work_authorization_complete: bool
    hiring_plan_details_role_and_purpose_count: int
    hiring_plan_details_time_and_urgency_count: int
    hiring_plan_details_work_authorization_count: int

    intent: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    agent_scratchpad: str
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]


# ─── Helper for initial state ─────────────────────────────────────────────────
def build_initial_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "interrupt_count": 0,
        "user_inputs": [],
        "messages": [],
        "user_messages": [],
        "job_details": {},
        "generated_job_descriptions": [],
        "additional_drafts" : False,
        "required_fields": [
                {"job_title": "Job Title"},
                {"skills": "Skills"},
                {"pay_range": "Pay Range"},
                {"start_date": "Start Date"},
                {"education": "Educational Requirements"},
            ],
        "job_details_missing": True,
        "hiring_support_option": None,
        "hiring_plan_details_role_and_purpose": "",
        "hiring_plan_details_time_and_urgency": "",
        "hiring_plan_details_work_authorization": "",
        "hiring_plan_details_role_and_purpose_complete": False,
        "hiring_plan_details_time_and_urgency_complete": False,
        "hiring_plan_details_work_authorization_complete": False,
        "hiring_plan_details_role_and_purpose_count": 0,
        "hiring_plan_details_time_and_urgency_count": 0,
        "hiring_plan_details_work_authorization_count": 0,
        "key_responsibilities": None,
        "intent": "",
        "agent_outcome": None,
        "agent_scratchpad": "",
        "intermediate_steps": [],
    }

def askLLM(query: str) -> str:
    try:
        prompt = system_query_prompt.format(
            input=query,
            agent_scratchpad=""
        )
        
        resp = llm.predict(prompt)
        return resp.strip()

    except Exception as e:
        print(f"[Error]: {str(e)}")


def get_validated_input(
    state: AgentState,
    field_name: str,
    prompts: list[str],
    llm_validation_prompt_template: str,
) -> AgentState:
    max_attempts = len(prompts)
    state["interrupt_count"] = 0

    for attempt in range(max_attempts):
        prompt = prompts[attempt]
        state["user_messages"].append(SystemMessage(content=prompt))

        user_input = interrupt(prompt)
        state["interrupt_count"] += 1

        validated_response = askLLM(
            llm_validation_prompt_template.format(user_input=user_input)
        ).strip()

        if validated_response.lower() != "no":
            state[field_name] = validated_response
            return state

    state["interrupt_count"] = 0
    return state


def get_hiring_plan_section(
    state: AgentState,
    section_key: str,
    message_body: str,
    success_message: str
) -> AgentState:
    """
    Generic collector for a ‘hiring plan’ section.
    - section_key: e.g. "role_and_purpose" or "time_and_urgency"
    - message_body: the full instructions to show the user
    - success_message: what to append when they answer validly
    """
    complete_k = f"hiring_plan_details_{section_key}_complete"
    count_k    = f"hiring_plan_details_{section_key}_count"
    data_k     = f"hiring_plan_details_{section_key}"
    
    # initialize counters & flags
    state.setdefault(count_k, 0)
    state[complete_k] = False
    
    # choose intro vs retry prefix
    prefix = (
        ""
        if state[count_k] == 0
        else "**It seems that the last response was not valid.** Please try again with more details.\n\n"
    )
    state[count_k] += 1
    
    # build & send prompt
    prompt = prefix + message_body
    state["user_messages"].append(AIMessage(content=prompt))
    user_input = interrupt(prompt)
    
    # validate via LLM
    decision = askLLM(
        f"I asked:\n{message_body}\n\n"
        f"User replied:\n{user_input}\n\n"
        "Is this userresponse relevant as per the question? Reply 'yes' or 'no' only."
    ).strip().lower()
    
    if decision == "yes":
        # store their answer
        state.setdefault(data_k, "")
        state[data_k] += user_input + " "
        state[complete_k] = True
        state["user_messages"].append(AIMessage(content=success_message))
    
    return state

# ─── State transition functions ────────────────────────────────────────────────
def run_agent(state: AgentState) -> AgentState:
    outcome = agent_runnable.invoke(state)
    state["agent_outcome"] = outcome

    if isinstance(outcome, AgentAction):
        name = outcome.tool
        result = tools_map[name].invoke(outcome.tool_input)
        #state["intermediate_steps"].append((outcome, str(result)))

        if name == "classify_intent":
            state["intent"] = result.strip().lower()
        else:
            state["user_messages"].append(AIMessage(content=str(result)))

        state["agent_scratchpad"] += f"Executed tool: {name}, Result: {result}\n"
        return state

    if isinstance(outcome, AgentFinish):
        state["user_messages"].append(AIMessage(content=outcome.return_values["output"]))
        return state

    return state


def handle_general_query(state: AgentState) -> AgentState:
    prompt = agent_prompt.format(
        input=state["input"],
        agent_scratchpad=state.get("agent_scratchpad", "")
    )
    resp = llm.predict(prompt)
    state["user_messages"].append(AIMessage(content=resp))
    return state

def handle_hiring_query(state: AgentState) -> AgentState:
   
    state["hiring_support_option"] = None #resetting
    input = state["input"]
    
    prompt = (
        "User has requested for hiring support and we need to help"
        f"User has asked the following {input} please take the user under confidence"
        f"Tell them that there are three ways: {HR_ACTIONS} Give a ordered list in same order"
    )

    intro_message = askLLM(prompt).strip()
    state["user_messages"].append(AIMessage(content=intro_message))

    option_chosen = interrupt("Choose the option")

    prompt = f"User had options {HR_ACTIONS} and has chosen '{option_chosen}'. Give me the number only that the user has chosen"
    option_chosen_by_user = int(askLLM(prompt).strip())
    state["hiring_support_option"] = option_chosen_by_user

    return state


def profile_match(state: AgentState) -> AgentState:
    count_of_generated_descriptions = len(state["generated_job_descriptions"])
    
    if count_of_generated_descriptions > 0:
        resp = "Please note: I will will only compare similarity with the last generated job description. **Please provide the candidate profile or resume as text.**"
        state["user_messages"].append(AIMessage(content=resp))
        candidate_profile = interrupt("Please provide candidate profile")
        similarity_data = {
                "job_description": state["generated_job_descriptions"][-1],
                "user_profile": candidate_profile
        }

        result = tools_map["match_profile_to_job"].invoke(json.dumps(similarity_data))
        resp = f"The candidate has a matching score of {result}% with the last generated job description."
        state["user_messages"].append(AIMessage(content=resp))

    else:
        resp = "It seems you have not generated a Job Description Yet. Please generate one. I can only compare score with the last generated Job Description"   
        state["user_messages"].append(AIMessage(content=resp))

    return state

def collect_job_details(state: AgentState, max_attempts: int = 5) -> AgentState:
    required_fields = state["required_fields"]

    ordered_list = "\n".join(
        f"{i+1}. {list(d.values())[0]}"
        for i, d in enumerate(required_fields)
    )


    # Build the question
    if not state.get("job_details"):
        question = (
            f"I can help you with writing a job description "
            "but please help me with these details:\n"
            f"{ordered_list}"
        )
    else:


        fields = [(k, v) for entry in required_fields for k, v in entry.items()]
        missing_labels = [label for key, label in fields if label not in  state.get("job_details")]
        if not missing_labels:
            state["job_details_missing"] = False 
            return state

        numbered_bullets = "\n".join(
                f"{i+1}. {label}"
                for i, label in enumerate(missing_labels)
        )

        question = (
            "Some details are still missing. Please provide the following:\n\n"
            f"{numbered_bullets}"
        )

    # Ask the user and record the prompt
    state["user_messages"].append(AIMessage(content=question))

    #####Interrupt Here####
    user_input = interrupt(question)


    # 3) Extract whatever fields they gave us
    extract_prompt = (
        f"These are required fields:\n{ordered_list}\n\n"
        f"We have these state data: {state['job_details']}"
        f"The user has input this: {user_input}"
        "Please extract data from the user input and map to the data fields missing in state data"
        "Only for keys for which data can be mapped and return as a JSON String"
    )

    extracted = askLLM(extract_prompt).strip()

    parsed: Dict[str, Any] = json.loads(extracted)
    job_details: Dict[str, str] = {}

    for key, value in parsed.items():
        if isinstance(value, list):
            # join lists into comma-separated strings
            job_details[key] = ", ".join(value)
        else:
            job_details[key] = str(value)

        
    state["job_details"].update(job_details)

    return state

def generate_job_description(state: AgentState) -> AgentState:
    # 1) Aggregate details once
    details = " ".join(f"{k}: {v}" for k, v in state.get("job_details", {}).items())

    # 2) Decide which focus we want
    is_second_draft = state.get("additional_drafts", False)
    focus = "technical skills" if is_second_draft else "cultural fit"

    # 3) Build a single prompt template
    prompt = (
        f"For {COMPANY_DETAILS['NAME']} in {COMPANY_DETAILS['LOCATION']}, "
        f"with mission “{COMPANY_DETAILS['MISSION']}” and clients {COMPANY_DETAILS['CLIENTS']}, "
        f"create a job description focused on {focus}, given these details: {details}. "
        f"Contact HR at {COMPANY_DETAILS['EMAIL']}."
    )

    # 4) Get the draft and append to messages
    draft = llm.predict(prompt)

    draft += ("\n\n**We are done generating drafts, please initiate the process again if you need to do any other task.**" if is_second_draft else "") 
    state["generated_job_descriptions"].append(draft)


    message = f"Here is your {focus} draft:\n\n{draft}\n\n"
    if not is_second_draft:
        message += "**Would you like another draft with emphasis on tech skills?**"
    state["user_messages"].append(AIMessage(content=message))

    # 5) If this was the first draft, ask whether to loop for a second
    if not is_second_draft:
        answer = interrupt("Do you need another draft?")
        decision = askLLM(
            f"I have asked user if they need another job draft and they have replied: '{answer}'. "
            "Interpret the user response as 'yes' or 'no' only"
        ).lower().strip()
        need_more = (decision == "yes")
        state["additional_drafts"] = need_more
        if not need_more:
            message= "Seems you are not interested in another draft. Let me know what else I can help you with."
            state["job_details"] = {} #Clearing up so that user can generate novel job detail if triggered again
            state["user_messages"].append(AIMessage(content=message))
    else:
        # we only allow two drafts, so reset the flag
        state["additional_drafts"] = False
    return state


def get_hiring_plan_role_and_purpose(state: AgentState) -> AgentState:
    body = (
        f"Okay,\nI see you’re planning to hire a new team member for **{COMPANY_DETAILS['NAME']}**. "
        "To help me find the best fit, please describe in one paragraph:\n\n"
        " • **The role you’re looking to fill** (job title & key responsibilities),\n"
        " • **The main purpose of this hire** (why it’s needed),\n"
        " • **What you hope to achieve** (outcomes or impact).\n\n"
        "Your detailed paragraph will allow us to tailor our hiring plan precisely to your goals. Thank you!"
    )
    return get_hiring_plan_section(
        state,
        section_key="role_and_purpose",
        message_body=body,
        success_message="That seems great! Proceeding further."
    )


def get_hiring_plan_time_and_urgency(state: AgentState) -> AgentState:
    body = (
        "Nice!!\nNeed more detail to understand the **time & urgency** of this hire:\n\n"
        " • **Urgency to fill**: How urgent is it to fill this vacancy?\n"
        " • **Preferred Start**: Immediate vs. phased onboarding?\n"
        " • **Hiring deadline**: Max time you have to fill the role.\n\n"
        "Once we know the urgency, we can plan next steps. Thank you!"
    )
    return get_hiring_plan_section(
        state,
        section_key="time_and_urgency",
        message_body=body,
        success_message="This seems good — let’s move ahead."
    )


def get_hiring_plan_work_authorization_requirements(state: AgentState) -> AgentState:
    
    body = (
        "Wonderful!!\nNow need details to know the work authorization requirments:\n\n"
        " • **Visa requirements**: are you willing to sponsor work visa?\n"
        " • Is there any visa category you are not willing to sponsor\n"
        "Once we know the visa requirement, it will help us to filter possible candidates. Thank you!"
    )
    return get_hiring_plan_section(
        state,
        section_key="work_authorization",
        message_body=body,
        success_message="Almost Done. Lets generate results"
    )

def generate_hiring_checklist(state: AgentState) -> AgentState:

    role_details      = state["hiring_plan_details_role_and_purpose"]
    urgency_details   = state["hiring_plan_details_time_and_urgency"]
    visa_requirements = state["hiring_plan_details_work_authorization"]

    prompt = (
        f"We have following details: {role_details}, {urgency_details}, {visa_requirements}" 
        f"We have to generate a hiring check list for company:{COMPANY_DETAILS['NAME']} having a mission: {COMPANY_DETAILS['MISSION']} having clients {COMPANY_DETAILS['CLIENTS']}, "
        f"We should give the timeline, tools we can use to make applications, social media approach, usage of AI tools, techniques to shortlist, etc"
        f"Use your imagination if there is something important that has to included and give checklist of minimum 5 points"
    )

    checklist = llm.predict(prompt)
    state["hiring_plan_details_role_and_purpose"]   = "" #resetting state params
    state["hiring_plan_details_time_and_urgency"]   = ""
    state["hiring_plan_details_work_authorization"] = ""
    state["user_messages"].append(AIMessage(content=checklist))
    
    return state


