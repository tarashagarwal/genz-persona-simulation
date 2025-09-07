# GenZ Persona Simulation

This project builds a **persona-aware reaction simulation engine** based on blog data, emotions, and sentiment. The idea is to simulate how different Gen‑Z personas might react—both internally and publicly—to a new piece of text.

---

## 📖 Overview

Most existing datasets lack **user reactions, emotions, and demographic data** together. To address this, we:

1. **Dataset**
   - Started with the **Blog Authorship Corpus**: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm  
     (mirror link in case of query params: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)
   - It includes blogger **age, job/industry, zodiac sign**, blog **text**, and **posting date**.
   - We derive **Gen‑Z classification** (was the author Gen‑Z at the time of posting?) from `post_date` + `age`.
   - We hypothesize potential **patterns between zodiac signs and expressed emotion/sentiment**.

2. **Emotion & Sentiment Augmentation**
   - **Emotions** → Google’s **GoEmotions** (27 fine‑grained labels)  
     - Blog post: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/  
     - Model used: `SamLowe/roberta-base-go_emotions` → https://huggingface.co/SamLowe/roberta-base-go_emotions
     - We group labels into:
       ```python
       POS_EMOS = {"admiration","amusement","gratitude","joy","love","optimism","pride","relief","caring","excitement"}
       NEG_EMOS = {"anger","annoyance","disappointment","disapproval","disgust","embarrassment","fear","grief","nervousness","remorse","sadness","confusion"}
       NEU_EMOS = {"neutral","realization","curiosity","surprise"}
       ```
   - **Sentiment** → Reddit‑trained sentiment classifier  
     - Model used: `minh21/XLNet-Reddit-Sentiment-Analysis` → https://huggingface.co/minh21/XLNet-Reddit-Sentiment-Analysis

3. **Masking Flag**
   - A heuristic that indicates if a writer might be **masking** feelings (emotion–sentiment **mismatch**), e.g. top emotion ∈ POS but sentiment ≤ 0, or top emotion ∈ NEG but sentiment ≥ 0.

4. **Processed Dataset on Hugging Face**
   - Public dataset: https://huggingface.co/datasets/tarashagarwal/genz-persona-simulation
   - Processing code:
     - `data_processing_code/BuildBlogsData.py` — converts raw blog dump → canonical CSV
     - `data_processing_code/BuildBlogDataWithSentiments.py` — adds GoEmotions, Reddit sentiment, Gen‑Z flag, masking

---

## 🧩 Persona Discovery (`data_processing_code/get_personas.py`)

This script discovers personas by clustering rows using text + categorical + emotion/sentiment features. It produces **assignments, a summary “persona card” JSON, and a PCA plot**.

### What the script does (step‑by‑step)

1. **Resolve input CSV**
   - Looks for `clusters_personas_genz_only.csv` in CWD or `/mnt/data`, or accept `--csv` path.
2. **Build emotion table**
   - Scans columns like `emo{n}_label` and `emo{n}_conf` to create a **wide** table of emotion confidences per row (`emotion_{label} = max(conf)`), joined back to the main DF.
3. **Sentiment score**
   - Converts `reddit_sentiment` strings → numeric `sentiment_score` (`positive` = +1, `negative` = −1, else 0).
4. **Feature construction**
   - **Text**: TF‑IDF (up to 5K terms, 1–2 grams) → **Truncated SVD (50 comps)** → `text_svd_0..49`
   - **Categorical**: One‑hot for `job`, `horoscope`, `top_emotion` (with unknowns ignored safely)
   - **Numeric**: `sentiment_score` and all `emotion_*` columns from step 2
   - A `ColumnTransformer` + `StandardScaler(with_mean=False)` compose the full feature space.
5. **K selection & clustering**
   - Tries **KMeans** for k ∈ [2..10] and picks the **best silhouette**. Falls back to k=2 if needed.
6. **Cluster descriptors**
   - Re‑vectorizes text (plain TF‑IDF) to extract **top distinguishers per cluster** (mean TF‑IDF ranking).
7. **Importance analysis**
   - Computes **mutual information** and **permutation importance** to see which features help separate clusters.
8. **Outputs**
   - `persona_assignments.csv` (original rows + `persona_id`)
   - `persona_cards.json` (sizes, shares, top words, top emotions, zodiac/job histograms, importance rankings)
   - `persona_pca_scatter.png` (2‑D PCA projection of the dense feature space)

👉 PCA sample:  
![Persona PCA](https://github.com/user-attachments/assets/938b78dd-7cb7-4dac-bbdf-85e0aaf7fc2c)

---

## 🔍 Persona‑Aware Retrieval (FAISS + LLM Reactions)

- Build **four FAISS indexes** — one per discovered persona — using the (SENTENCE) embeddings of each comment’s text.
- At runtime, when a user provides input and selects a persona:
  1. **Search** that persona’s FAISS index, get top match and similarity score.
  2. If score ≥ threshold, pull the matched row’s **emotions, sentiment, masking flag**.
  3. Call the LLM to generate **two outputs**:
     - **Public reaction**: conditioned on emotions + sentiment + masking (how they’d speak publicly).
     - **Internal reaction**: conditioned only on emotions (private thought; ignores masking/sentiment).

> This avoids leaking matched text to the LLM; only **attributes** are passed once similarity clears a threshold.

---

## 📊 Persona Summary (from `persona_cards.json`)

### Dataset Metadata

| Attribute | Value |
|---|---|
| **Source CSV** | `/home/tarash/Personal/genz-persona-simulation/clusters_personas_genz_only.csv` |
| **Rows** | `96,639` |
| **Personas** | `4` |
| **Silhouette Score** | `0.18065077133961818` |

### Feature Importance — Mutual Information (Top 30)

| # | Feature | Score |
|---:|---|---:|
| 1 | top_emotion_neutral | 0.5428501714127949 |
| 2 | emotion_neutral | 0.5320947797263818 |
| 3 | emotion_approval | 0.2566405583966427 |
| 4 | sentiment_score | 0.24942562047139494 |
| 5 | emotion_annoyance | 0.22132267656667826 |
| 6 | emotion_realization | 0.21773726724569875 |
| 7 | text_svd_4 | 0.21274953438184463 |
| 8 | text_svd_1 | 0.19993548266429317 |
| 9 | text_svd_9 | 0.1900586474149244 |
| 10 | text_svd_2 | 0.1822014123268676 |
| 11 | emotion_joy | 0.17988654946481253 |
| 12 | text_svd_0 | 0.17860431315548064 |
| 13 | emotion_admiration | 0.17460422591334668 |
| 14 | emotion_love | 0.17382299871345408 |
| 15 | emotion_amusement | 0.1660951189182951 |
| 16 | text_svd_8 | 0.16553017787984303 |
| 17 | text_svd_7 | 0.15994895882878724 |
| 18 | text_svd_3 | 0.15531168005459794 |
| 19 | text_svd_18 | 0.15141827855484324 |
| 20 | top_emotion_love | 0.15098223793711507 |
| 21 | text_svd_13 | 0.14990047030248 |
| 22 | text_svd_17 | 0.1473581273308937 |
| 23 | text_svd_5 | 0.1399305130106956 |
| 24 | text_svd_11 | 0.13483987419287335 |
| 25 | text_svd_10 | 0.13311187786613887 |
| 26 | text_svd_19 | 0.13306506096149628 |
| 27 | text_svd_36 | 0.1312698981824183 |
| 28 | text_svd_39 | 0.13059938942389038 |
| 29 | text_svd_34 | 0.13039000637995435 |
| 30 | top_emotion_amusement | 0.14815528298590142 |

> Note: The JSON you provided lists 30 items; the table preserves them (ordering by reported score).

### Feature Importance — Permutation Importance (Top 30)

| # | Feature | Importance |
|---:|---|---:|
| 1 | top_emotion_neutral | 0.007553886112232108 |
| 2 | top_emotion_love | 0.00043590061983261874 |
| 3 | emotion_neutral | 0.0002793903082606297 |
| 4 | emotion_love | 0.000028456420285816186 |
| 5 | sentiment_score | 0.00001034778919484225 |
| 6 | emotion_nervousness | 0.00001034778919484225 |
| 7 | text_svd_0 | 0.0 |
| 8 | text_svd_1 | 0.0 |
| 9 | text_svd_2 | 0.0 |
| 10 | text_svd_3 | 0.0 |
| 11 | text_svd_4 | 0.0 |
| 12 | text_svd_5 | 0.0 |
| 13 | text_svd_6 | 0.0 |
| 14 | text_svd_7 | 0.0 |
| 15 | text_svd_8 | 0.0 |
| 16 | text_svd_9 | 0.0 |
| 17 | text_svd_10 | 0.0 |
| 18 | text_svd_11 | 0.0 |
| 19 | text_svd_12 | 0.0 |
| 20 | text_svd_13 | 0.0 |
| 21 | text_svd_14 | 0.0 |
| 22 | text_svd_15 | 0.0 |
| 23 | text_svd_16 | 0.0 |
| 24 | text_svd_17 | 0.0 |
| 25 | text_svd_18 | 0.0 |
| 26 | text_svd_19 | 0.0 |
| 27 | text_svd_20 | 0.0 |
| 28 | text_svd_21 | 0.0 |
| 29 | text_svd_22 | 0.0 |
| 30 | text_svd_23 | 0.0 |

---

### Personas

#### Persona 0
| Field | Value |
|---|---|
| **Size** | 3395 |
| **Share** | 0.03513074431647679 |
| **Top Words** | love, you, love you, and, my, to, the, me, it, that, so, in |
| **Avg Sentiment Score** | 0.9802650957290132 |
| **Top Emotions (avg_conf)** | love (0.9335), admiration (0.0530), approval (0.0346), desire (0.0261), joy (0.0254), optimism (0.0235), curiosity (0.0131), neutral (0.0129) |
| **Horoscope Top** | Leo, Virgo, Taurus, Aries, Gemini |
| **Job Top** | indUnk, Student, Arts, Technology, Education |

#### Persona 1
| Field | Value |
|---|---|
| **Size** | 17429 |
| **Share** | 0.18035161787684062 |
| **Top Words** | the, to, and, it, you, my, that, of, for, was, is, so |
| **Avg Sentiment Score** | 0.8615525847725056 |
| **Top Emotions (avg_conf)** | amusement (0.4066), gratitude (0.3103), admiration (0.1448), optimism (0.0836), joy (0.0688), confusion (0.0345), approval (0.0257), neutral (0.0229) |
| **Horoscope Top** | Aries, Libra, Taurus, Scorpio, Virgo |
| **Job Top** | indUnk, Student, Technology, Arts, Education |

#### Persona 2
| Field | Value |
|---|---|
| **Size** | 58 |
| **Share** | 0.0006001717733006343 |
| **Top Words** | scared, afraid, scary, of, the, my, to, is, and, am, me, it |
| **Avg Sentiment Score** | -0.5172413793103449 |
| **Top Emotions (avg_conf)** | fear (0.8980), nervousness (0.1070), neutral (0.0407), confusion (0.0246), sadness (0.0151), curiosity (0.0028), disgust (0.0024), realization (0.0023) |
| **Horoscope Top** | Aries, Aquarius, Cancer, Taurus, Leo |
| **Job Top** | Student, indUnk, Education, Technology, Law |

#### Persona 3
| Field | Value |
|---|---|
| **Size** | 75757 |
| **Share** | 0.783917466033382 |
| **Top Words** | urllink, nbsp, the, nbsp urllink, and, of, to, in, is, on, for, more |
| **Avg Sentiment Score** | 0.12033211452407039 |
| **Top Emotions (avg_conf)** | neutral (0.9535), approval (0.0202), realization (0.0079), annoyance (0.0067), disapproval (0.0012), excitement (0.0009), confusion (0.0008), admiration (0.0006) |
| **Horoscope Top** | Aries, Virgo, Leo, Gemini, Taurus |
| **Job Top** | indUnk, Student, Technology, Education, Arts |

---

## 🖥️ Application Setup

### Prerequisites
- **Node.js** v20.19.1 (LTS) — includes `npm`
- **yarn** v1.22.22
- **Python** 3.10.9 + **pip** 25.1.1
- **OpenAI API key** in `genz-persona-simulation/agentic_logic/.env`:
  ```dotenv
  OPENAI_API_KEY=sk-...
  ```

### Frontend (Next.js)
```bash
# from repo root
yarn
yarn dev
# http://localhost:3000
```

### Backend (Flask + Agent Logic)
```bash
# from repo root
cd agentic_logic
pip install -r requirements.txt
python -m agent.HRAgent
# http://localhost:5000
```

### Credentials (dev)
Any valid email + password that fits format will work. Example:
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

## 📸 Screenshots

- **Landing Page**  
  https://github.com/user-attachments/assets/5ed2fd6d-c05b-4531-aa04-916cedf17dac

- **User Output**  
  https://github.com/user-attachments/assets/1af82ab4-932c-422d-97cd-9ba180c17852

---

## 🔗 Key Links

- Blog Authorship Corpus: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
- GoEmotions blog: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
- GoEmotions model (HF): https://huggingface.co/SamLowe/roberta-base-go_emotions
- Reddit Sentiment model (HF): https://huggingface.co/minh21/XLNet-Reddit-Sentiment-Analysis
- Processed dataset (HF): https://huggingface.co/datasets/tarashagarwal/genz-persona-simulation

---

## 📦 Repo Structure (high‑level)

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
├─ frontend/ (if applicable)   # Next.js app or root-level Next app
├─ persona_assignments.csv      # (generated)
├─ persona_cards.json           # (generated)
├─ persona_pca_scatter.png      # (generated)
└─ README.md
```

---

## 🚀 Roadmap Ideas

- Add more demographic‑rich sources (e.g., PANDORA) when licensing allows.
- Improve **masking** detection with pragmatic/context cues.
- Persona **drift** tracking over time.
- Live demo + telemetry on retrieval quality.

---

## 📝 License & Attribution

- Respect the licenses/terms of the Blog Authorship Corpus and referenced models/datasets.
- Cite Google GoEmotions and the respective HF model authors.
- This repository provides processing and modeling code; **source datasets may require separate download/consent**.