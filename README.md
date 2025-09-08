# Gen‑Z Persona Simulation

![Landing Page](https://github.com/user-attachments/assets/5ed2fd6d-c05b-4531-aa04-916cedf17dac)
![User Output](https://github.com/user-attachments/assets/1af82ab4-932c-422d-97cd-9ba180c17852)

**What if you could ask:** *“How would different types of Gen‑Z react to this?”*  
This prototype lets you try that in seconds. Type a message, pick a persona, and get two perspectives:
- **Public Reaction** — how they’d likely respond out loud
- **Internal Thought** — what they might actually feel inside

It’s simple, fast, and fun—built from real blog posts, emotion & sentiment models, light clustering, and a tiny retrieval + LLM layer.

---

## ✨ TL;DR — How it feels to use

1. Open the app, **choose a persona** (e.g., Love, Surprise, Fear, Neutral vibes).  
2. **Paste any statement** (news headline, tweet, your scenario).  
3. Click **Find Reaction**.  
4. See the persona’s **Public** vs **Private** reactions side‑by‑side...grounded by similar past posts we found.

> We don’t send matched text to the LLM...only attributes (emotions/sentiment/masking) once similarity is high—so the reaction stays persona‑aligned without copying anyone’s text.

---

## 🎨 The Frontend (single, friendly dashboard)

We keep the UI minimal so anyone can play with the idea quickly.

- **Live code (public):**  
  https://github.com/tarashagarwal/genz-persona-simulation/blob/main/src/app/auth/(dashboard)/dashboard/page.tsx

**What’s on the page**
- A **persona picker** (4 clusters discovered from data)
- A **text box** for your input
- A **Find Reaction** button to call the backend agent
- Two result cards:
  - **Public Reaction** (emotions + sentiment + masking applied)
  - **Private Reaction (Internal Thought)** (emotions only)
- Optional metadata: **similarity score**, **top emotions**, and whether **masking** seems likely

> The dashboard is intentionally one page..no maze of routes. Paste text, pick persona, get reactions.

---

## 🧠 How it works (descriptive version)

- We start with the **Blog Authorship Corpus** (Blogger.com): age, job, zodiac, post date, and text.  
  From *date + age*, we infer if the author would be **Gen‑Z** at the time of posting.
- We enrich each post with **emotions** via **GoEmotions (RoBERTa)** and **sentiment** via a **Reddit‑trained XLNet** model.
- If emotion and sentiment disagree, we flag it as **masking** (a realistic “I look fine, I feel awful” pattern).
- We then discover **4 personas** by clustering primarily on emotional tendencies.
- Finally, we build **four FAISS indexes**...one per persona. When you type something, we:
  1) search the chosen persona’s index,
  2) if similarity is high, 
  3) pull its attributes (emotions/sentiment/masking), and 
  4) ask an LLM to write both the **Public** and **Internal** reactions in that style.

That’s it...no massive training jobs, just smart reuse of signals already in the data.

---

## 🧪 Gen‑Z Persona Simulation ... Design Notes

### Data Sources & Rationale
- We initially aimed for **PANDORA Talks** (demographics + personality): https://arxiv.org/pdf/2004.04460  
  Access required approval; we’ve requested it and may update later.
- We used the **Blog Authorship Corpus** instead: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm  
  Helpful fields: **post date**, **age** (to infer Gen‑Z), **job**, and **zodiac**.
- To enrich the text we used:
  - **GoEmotions (RoBERTa‑base)**: https://huggingface.co/SamLowe/roberta-base-go_emotions
  - **XLNet Reddit Sentiment**: https://huggingface.co/minh21/XLNet-Reddit-Sentiment-Analysis

This gave us a workable dataset for clustering and persona design.

### Modeling Choice & Justification
We didn’t train a new model from scratch (limited time & data). Instead, we:
1) **Clustered** posts into personas using emotion‑forward features,  
2) **Embedded** comments and stored them per‑persona in **FAISS**,  
3) **Matched** new inputs by vector similarity (with a cutoff),  
4) **Generated** persona‑aligned reactions using an LLM (e.g., *gpt‑4.1‑mini*) with only the attributes.

Why this route?
- Fine‑tuning wasn’t feasible quickly.  
- Vector similarity worked well in early trials.  
- It keeps infra light while still feeling persona‑aware.

### Persona Design Assumptions
- Personas largely reflect **emotional tendencies** (Love, Surprise, Fear, Neutral).  
- Zodiac/job/writing style didn’t produce strong clusters here.  
- **Emotion vs Sentiment can diverge**; we treat that as **masking**.

### Ethical & Bias Considerations
- **LLM tone** may soften strong language...true Gen‑Z speech can be spicier.  
- **Zero‑shot** models may miss slang/cultural nuance...fine‑tuning would help.  
- **Sampling bias**: Blogger.com ≠ a perfect Gen‑Z mirror.

### Scaling Path to a Full Product
**Data & Training**
- Persona sizes: 3,395 • 17,429 • 58 (sparse) • 75,757  
- Needs: bigger, fresher Gen‑Z text; balanced clusters; regular re‑indexing.

**Infrastructure**
- Cloud **GPUs** for faster indexing/refresh.  
- Automated pipelines for **fetch → enrich → cluster → index**.  
- **Cost per request** tracking; start serverless, scale to GPU nodes if needed.

**UX & Testing**
- Define a test set + **confidence threshold** (≥85%).  
- A/B test whether reactions feel **relevant**, **authentic**, and **engaging**.

**Summary**
Vector search + LLMs can simulate persona reactions **today** with modest resources.  
To ship it as a product, add: **larger balanced datasets**, **Gen‑Z‑aware fine‑tuning**, **solid retraining pipelines**, and **ethics & trust** checks.

---

## 📊 Snapshot of the Data & Personas

- **Rows processed**: 96,639  
- **Discovered personas**: 4  
- **Vibe summary** (informal):
  - **P0 — Love‑forward**: affectionate, upbeat, warm
  - **P1 — Amusement/Gratitude**: playful, appreciative
  - **P2 — Fear/Nervous**: anxious, sensitive (very small cluster)
  - **P3 — Neutral/Info‑sharing**: mostly matter‑of‑fact posts

> Full technical breakdown (silhouette, feature importances, per‑persona stats) lives in the generated artifacts: `persona_cards.json`, `persona_assignments.csv`, `persona_pca_scatter.png`.

---

## 🛠️ Setup (3 minutes)

### Prerequisites
- **Node.js** v20.19.1 (LTS) + **npm**
- **yarn** v1.22.22
- **Python** 3.10.9 + **pip** 25.1.1
- **OpenAI API key** in `genz-persona-simulation/agentic_logic/.env`
  ```dotenv
  OPENAI_API_KEY=sk-...
  ```

### 1) Start the Frontend (Next.js)
```bash
yarn
yarn dev
# http://localhost:3000
```

### 2) Build Indexes
```bash
cd agentic_logic
python build_index.py
```

### 3) Start the Backend (Flask + Agent Logic)
```bash
cd agentic_logic
pip install -r requirements.txt
python -m agent.HRAgent
# http://localhost:5000
```

### 4) Log in (dev)
Any valid email/password *format* works. Example:
```
Email:    test@gmail.com
Password: test12345678
```

### Components & Ports
| Component | Command | Port |
|---|---|---|
| Next.js Frontend | `yarn dev` | 3000 |
| Flask HR Agent | `python -m agent.HRAgent` | 5000 |

---

## 🔗 Key Links
- Blog Authorship Corpus — https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm  
- GoEmotions blog — https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/  
- GoEmotions model (HF) — https://huggingface.co/SamLowe/roberta-base-go_emotions  
- Reddit Sentiment model (HF) — https://huggingface.co/minh21/XLNet-Reddit-Sentiment-Analysis  
- Processed dataset (HF) — https://huggingface.co/datasets/tarashagarwal/genz-persona-simulation  
- Frontend page (public) — https://github.com/tarashagarwal/genz-persona-simulation/blob/main/src/app/auth/(dashboard)/dashboard/page.tsx

---

## 🗂️ Repo Structure (high‑level)
```
genz-persona-simulation/
├─ agentic_logic/
│  ├─ agent/
│  │  └─ HRAgent.py            # Flask app entry (python -m agent.HRAgent)
│  └─ .env                     # OPENAI_API_KEY=...
├─ data_processing_code/
│  ├─ BuildBlogsData.py
│  ├─ BuildBlogDataWithSentiments.py
│  └─ get_personas.py
├─ persona_assignments.csv      # (generated)
├─ persona_cards.json           # (generated)
├─ persona_pca_scatter.png      # (generated)
└─ README.md
```

---