# search_and_match.py
# ---------------------------------------------------------
# Persona-aware FAISS search + optional OpenAI generation.
# - Uses constants.py for all config.
# - Avoids matched-text leakage; only attributes sent when similarity >= threshold.
# - Returns emotions, masking (bool), reddit_sentiment, score, and TWO LLM texts:
#   public_reaction (sentiment+masking-aware) and private_reaction (emotions-only).
# ---------------------------------------------------------

from __future__ import annotations
import os, json, argparse
from pathlib import Path
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pprint import pformat

# 🔹 import your constants module
import constants as C

# ---------- Helpers ----------
def load_persona_card(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        card = json.load(f)
    if "personas" not in card:
        raise ValueError("Persona card JSON missing 'personas' key.")
    return card

def load_index_and_meta(persona_id: int):
    idx_path = C.INDEX_DIR / f"persona_{persona_id}.faiss"
    meta_path = C.INDEX_DIR / f"persona_{persona_id}.meta.parquet"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing index/meta for persona {persona_id} in {C.INDEX_DIR}")
    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)
    return index, meta

def extract_high_conf_emotions(row_dict: dict, threshold: float = C.EMO_CONF_THRESHOLD):
    out = []
    for i in range(1, 5):
        lab = row_dict.get(f"emo{i}_label")
        conf = row_dict.get(f"emo{i}_conf")
        if isinstance(lab, str) and pd.notna(conf) and float(conf) >= threshold:
            out.append({"label": lab, "confidence": float(conf)})
    return out

def drop_text_and_lowconf_emotions(row_dict: dict, threshold: float = C.EMO_CONF_THRESHOLD):
    copy = dict(row_dict)
    copy.pop(C.TEXT_COL, None)  # never leak matched text
    for i in range(1, 4 + 1):
        lk, ck = f"emo{i}_label", f"emo{i}_conf"
        lab, conf = copy.get(lk), copy.get(ck)
        keep = isinstance(lab, str) and pd.notna(conf) and float(conf) >= threshold
        if not keep:
            copy.pop(lk, None); copy.pop(ck, None)
    return copy

def summarize_persona_traits(persona_card: dict, persona_id: int,
                             top_k_emotions: int = 5, top_k_words: int = 6) -> str:
    p = persona_card["personas"].get(str(persona_id))
    if not p:
        return f"Persona #{persona_id} traits: (no card found)"
    emos = [e["emotion"] for e in (p.get("top_emotions") or [])][:top_k_emotions]
    words = (p.get("top_words") or [])[:top_k_words]
    avg_sent = p.get("avg_sentiment_score", 0.0)
    tone = "positive" if avg_sent > 0.2 else ("negative" if avg_sent < -0.2 else "mixed/neutral")
    return (
        f"Persona #{persona_id} traits:\n"
        f"- Typical emotions: {', '.join(emos) if emos else 'n/a'}\n"
        f"- Common words/phrases: {', '.join(words) if words else 'n/a'}\n"
        f"- Overall tone tendency: {tone}\n"
    )

# 🔹 sentiment helpers
def _first_numeric(d: dict, keys: list[str]):
    for k in keys:
        if k in d and pd.notna(d[k]):
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None

def extract_reddit_sentiment_and_score(row_dict: dict):
    label = row_dict.get("reddit_sentiment") or row_dict.get("sentiment")
    score = _first_numeric(
        row_dict,
        ["reddit_sentiment_score", "reddit_score", "sentiment_score", "reddit_compound", "compound", "sentiment_compound"],
    )
    return label, score

# ---------- Prompt builder (outputs JSON with two reactions) ----------
def build_prompt(
    user_text: str,
    persona_traits: str,
    similarity: float,
    matched_attrs: dict | None,
    high_conf_emotions: list[dict] | None,
    *,
    masking: bool | None = None,
    sentiment_label: str | None = None,
    sentiment_score: float | None = None,
) -> list[dict]:
    """
    Builds messages so the model returns a JSON object:
      { "public_reaction": "...", "private_reaction": "..." }
    public_reaction follows persona + sentiment + masking (if high similarity);
    private_reaction ignores sentiment/masking and just expresses emotions.
    """
    sys_lines = [
        "You are a Gen Z individual in the United States.",
        "Return ONLY a valid JSON object with exactly two string fields: public_reaction and private_reaction.",
        "Both fields should be 1–2 sentences each.",
        "Use emojis and hashtags sparingly in public_reaction (0–2 emojis, at most one hashtag).",
        "Do not add extra keys, comments, or backticks.",
    ]

    # Emotions to convey (order already reflects confidence)
    if high_conf_emotions:
        emo_labels = ", ".join([str(e.get("label")) for e in high_conf_emotions if e.get("label")])
        if emo_labels:
            sys_lines.append(f"Target emotions to convey: {emo_labels} (in priority order).")

    # Overall sentiment (public only)
    if sentiment_label:
        if isinstance(sentiment_score, (int, float)):
            sys_lines.append(f"Overall sentiment to reflect in public_reaction: {sentiment_label} (score {sentiment_score:.2f}).")
        else:
            sys_lines.append(f"Overall sentiment to reflect in public_reaction: {sentiment_label}.")

    # Masking behavior (public only)
    if masking is True:
        sys_lines.append("Masking is TRUE for public_reaction: hedge and understate; avoid strong exclamations; prefer neutral/soft emojis (or none).")
    elif masking is False:
        sys_lines.append("Masking is FALSE for public_reaction: be clear and appropriately expressive; enthusiasm is okay when fitting.")

    system_content = "\n".join(sys_lines)

    # Context + user message
    ctx_lines = [persona_traits.strip()]
    if similarity >= C.SIM_THRESHOLD and matched_attrs:
        ctx_lines.append(f"Nearest-sample similarity: {similarity:.2f}")
        ctx_lines.append("Nearest-sample attributes (no text): " + json.dumps(matched_attrs, ensure_ascii=False))
        if high_conf_emotions:
            ctx_lines.append("Nearest-sample high-confidence emotions: " + json.dumps(high_conf_emotions, ensure_ascii=False))

    user_content = (
        "Message: " + user_text.strip() + "\n\n"
        "- Produce public_reaction that follows persona traits, target emotions, overall sentiment (if given), and masking rules.\n"
        "- Produce private_reaction that ignores sentiment and masking and expresses only the emotions (raw/unguarded), still in persona voice.\n"
        "- Output JSON only."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(ctx_lines) + "\n\n" + user_content},
    ]

# ---------- Main react ----------
def react(user_text: str, persona_id: int, persona_card_path: str | Path,
          openai_key: str | None, verbose: bool):
    def log(msg):
        if verbose:
            print(msg)

    # Load persona card & index
    log(f"[load] persona_card: {persona_card_path}")
    persona_card = load_persona_card(persona_card_path)

    log(f"[load] FAISS index & meta for persona {persona_id} from {C.INDEX_DIR}")
    index, meta = load_index_and_meta(persona_id)
    log(f"[meta] rows={len(meta)}, cols={list(meta.columns)}")

    # Embed + search
    log(f"[embed] model={C.EMB_MODEL}")
    emb_model = SentenceTransformer(C.EMB_MODEL)
    qvec = emb_model.encode([user_text], normalize_embeddings=True).astype("float32")
    log(f"[embed] qvec shape={qvec.shape}")

    D, I = index.search(qvec, 1)
    sim_raw = float(D[0, 0])
    sim = max(0.0, min(1.0, sim_raw))  # clamp to [0,1]
    log(f"[faiss] raw_score={sim_raw:.4f}  clamped_similarity={sim:.4f}  (threshold={C.SIM_THRESHOLD:.2f})")

    best_idx = int(I[0, 0])
    best_row = meta.iloc[best_idx].to_dict()
    log(f"[faiss] matched_row_index={best_idx}  matched_row_id={best_row.get(C.ID_COL)}")

    attrs_no_text = drop_text_and_lowconf_emotions(best_row, threshold=C.EMO_CONF_THRESHOLD)
    high_conf = extract_high_conf_emotions(best_row, threshold=C.EMO_CONF_THRESHOLD)
    reddit_label, reddit_score = extract_reddit_sentiment_and_score(best_row)

    log(f"[attrs] using_attributes={sim >= C.SIM_THRESHOLD}")
    if sim >= C.SIM_THRESHOLD:
        sent_keys = list(attrs_no_text.keys())
        log(f"[attrs] keys_sent (no text): {sent_keys}")
        log(f"[attrs] high_conf_emotions_sent (>= {C.EMO_CONF_THRESHOLD}): {high_conf}")

    traits = summarize_persona_traits(persona_card, persona_id)
    log("[persona] traits:\n" + traits)

    # Only include sensitive guidance when similarity is high
    masking_for_prompt = bool(int(best_row.get("masking", 0))) if sim >= C.SIM_THRESHOLD else None
    sentiment_label_for_prompt = reddit_label if sim >= C.SIM_THRESHOLD else None
    sentiment_score_for_prompt = reddit_score if sim >= C.SIM_THRESHOLD else None

    messages = build_prompt(
        user_text=user_text,
        persona_traits=traits,
        similarity=sim,
        matched_attrs=attrs_no_text if sim >= C.SIM_THRESHOLD else None,
        high_conf_emotions=high_conf if sim >= C.SIM_THRESHOLD else None,
        masking=masking_for_prompt,
        sentiment_label=sentiment_label_for_prompt,
        sentiment_score=sentiment_score_for_prompt,
    )
    log("[prompt] messages to OpenAI:\n" + pformat(messages, width=100, compact=False))

    # Prepare non-LLM outputs (avoid attribute leakage when low)
    if sim >= C.SIM_THRESHOLD:
        top_emotion_out = best_row.get("top_emotion")
        emotion_scores_out = high_conf
        masking_out = bool(int(best_row.get("masking", 0)))
        reddit_sentiment_out = reddit_label
        reddit_score_out = reddit_score
    else:
        top_emotion_out = None
        emotion_scores_out = []
        masking_out = False
        reddit_sentiment_out = None
        reddit_score_out = None

    # API key handling
    api_key = openai_key or C.get_openai_key()
    if not api_key:
        log("[warn] OPENAI_API_KEY missing -> skipping LLM call.")
        return {
            "persona_id": persona_id,
            "similarity": sim,
            "matched_row_id": best_row.get(C.ID_COL),
            "used_attributes": sim >= C.SIM_THRESHOLD,
            "top_emotion": top_emotion_out,
            "emotion_scores": emotion_scores_out,
            "masking": masking_out,
            "reddit_sentiment": reddit_sentiment_out,
            "score": reddit_score_out,
            "attributes_sent": attrs_no_text if sim >= C.SIM_THRESHOLD else None,
            "high_conf_emotions_sent": high_conf if sim >= C.SIM_THRESHOLD else None,
            "reaction": None,                 # kept for backward-compat (maps to public_reaction)
            "public_reaction": None,
            "private_reaction": None,
            "error": "OPENAI_API_KEY missing. Set it in .env or pass --openai-key.",
            "prompt_preview": messages,
        }

    # temperature: slightly lower when masking True to keep things restrained
    temperature = 0.35 if masking_for_prompt is True else 0.6

    log(f"[openai] model={C.GEN_MODEL} key_present=True temperature={temperature}")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=C.GEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=120,
        response_format={"type": "json_object"},   # ask for strict JSON
    )

    raw = resp.choices[0].message.content.strip()
    log("[openai] raw content:\n" + raw)

    # Parse JSON safely
    public_rx, private_rx = None, None
    try:
        obj = json.loads(raw)
        public_rx = (obj.get("public_reaction") or "").strip() or None
        private_rx = (obj.get("private_reaction") or "").strip() or None
    except Exception as e:
        log(f"[openai] JSON parse error: {e}")

    return {
        "persona_id": persona_id,
        "similarity": sim,
        "matched_row_id": best_row.get(C.ID_COL),
        "used_attributes": sim >= C.SIM_THRESHOLD,
        "top_emotion": top_emotion_out,
        "emotion_scores": emotion_scores_out,
        "masking": masking_out,                          # bool
        "reddit_sentiment": reddit_sentiment_out,
        "score": reddit_score_out,
        "attributes_sent": attrs_no_text if sim >= C.SIM_THRESHOLD else None,
        "high_conf_emotions_sent": high_conf if sim >= C.SIM_THRESHOLD else None,
        # New dual outputs:
        "public_reaction": public_rx,
        "private_reaction": private_rx,
        # For backward compatibility (UI already reads `reaction`)
        "reaction": public_rx,
    }

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", type=int, default=C.DEFAULT_PERSONA_ID,
                    help=f"persona_id to search (default: {C.DEFAULT_PERSONA_ID})")
    ap.add_argument("--text", type=str, default=C.DEFAULT_TEXT,
                    help=f"input text (default: '{C.DEFAULT_TEXT}')")
    ap.add_argument("--persona-card", type=str, default=C.PERSONA_CARD_PATH,
                    help=f"path to persona card JSON (default: {C.PERSONA_CARD_PATH})")
    ap.add_argument("--openai-key", type=str, default=None,
                    help="OpenAI API key (overrides .env/ENV)")
    ap.add_argument("--verbose", action="store_true",
                    default=C.VERBOSE_DEFAULT,
                    help="print detailed logs (default from VERBOSE env)")
    ap.add_argument("--no-verbose", dest="verbose", action="store_false")
    args = ap.parse_args()

    out = react(args.text, args.persona, args.persona_card, args.openai_key, args.verbose)
    print(json.dumps(out, indent=2, ensure_ascii=False))
