# app.py
import os, json, traceback
import numpy as np
from flask import Flask, request, jsonify

# If you want to call Flask directly from the browser (port 3000 -> 5000), enable CORS:
# from flask_cors import CORS

# --- your modules ---
from search_and_match import react
import constants as C  # must exist next to search_and_match.py
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["*"]}},
    supports_credentials=False
)

# Uncomment if NOT using Next.js rewrite proxy and you call Flask directly from 3000:
# CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

def _np_json(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    return str(o)

@app.post("/api/find-reaction")
def find_reaction():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        persona_id = int(data.get("persona_id", C.DEFAULT_PERSONA_ID))
        persona_card_path = data.get("persona_card") or C.PERSONA_CARD_PATH

        # react() will use OPENAI_API_KEY via constants.get_openai_key() if not provided
        openai_key = os.getenv("OPENAI_API_KEY")
        verbose_env = os.getenv("VERBOSE")
        verbose = (verbose_env == "1") if verbose_env is not None else bool(C.VERBOSE_DEFAULT)

        out = react(
            user_text=text,
            persona_id=persona_id,
            persona_card_path=persona_card_path,
            openai_key=openai_key,
            verbose=verbose,
        )

        return app.response_class(
            response=json.dumps(out, ensure_ascii=False, default=_np_json),
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        if os.getenv("DEBUG", "1") == "1":
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.get("/api/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    # helps some HF tokenizers avoid warning spam
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    app.run(host="0.0.0.0", port=5000, debug=True)
