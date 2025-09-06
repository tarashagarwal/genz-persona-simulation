import os
import random
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json

# Load environment so GPT_MODEL_NAME is available
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)

llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))

here = os.path.dirname(__file__)
hiring_intent_data_file = os.path.join(here, "data", "hiring_intent.txt")

@tool("classify_intent", return_direct=True)
def classify_intent(input_text: str) -> str:
    """Classify the user's intent into one of: hiring, general_query."""

    examples = []
    try:
        with open(hiring_intent_data_file, "r") as f:
            examples = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("⚠️ hiring_intent.txt file not found. Proceeding without examples.")

    examples_text = "\n".join(f"- {ex}" for ex in examples)

    prompt = f"""
You are an intent classifier. Read the following user input and classify the intent into one of these categories:
"hiring" or "general_query".

Examples of *hiring* intent:
{examples_text}

Now classify this user input:
"{input_text}"

Just return one of the two categories: hiring or general_query.
"""

    result = llm.invoke(prompt)
    return result.content.strip().lower()




from langchain.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.tools import tool
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import tool

@tool("match_profile_to_job", return_direct=False)
def match_profile_to_job(input: str) -> str:
    """
    Compares a candidate profile with a job description and returns a match score.

    **Input**: either
      - A JSON‐encoded string with two keys:
          {
            "job_description": "...",
            "user_profile": "..."
          }
      - A plain-text string in this form:
          Job description: <text>
          User profile: <text>

    **Returns**:
      - A percentage string (e.g. "87.65%") indicating how well the two texts match.
      - Or an error message if inputs are missing or malformed.
    """
    try:
        # Try to load as JSON first
        try:
            data = json.loads(input)
            jd = data.get("job_description", "").strip()
            profile = data.get("user_profile", "").strip()
        except json.JSONDecodeError:
            # Fallback: parse plain-text "Job description: ... User profile: ..."
            m = re.search(
                r"Job description:(.*?)User profile:(.*)",
                input,
                re.IGNORECASE | re.DOTALL
            )
            if not m:
                return (
                    "Invalid input format. "
                    "Provide either a JSON string with 'job_description' and 'user_profile' keys, "
                    "or plain text starting with 'Job description:' followed by 'User profile:'."
                )
            jd, profile = m.group(1).strip(), m.group(2).strip()

        if not jd or not profile:
            return "Job description or profile text is missing."

        # Vectorize and compute cosine similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([jd, profile])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # Return as percentage
        percent = round(sim * 100, 2)
        return f"{percent}"

    except Exception as e:
        return f"Error occurred: {str(e)}"


tools = [classify_intent, match_profile_to_job]
tools_map = {t.name: t for t in tools}