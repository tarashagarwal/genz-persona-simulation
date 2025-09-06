
# 🧠 AI-Powered HR Assistant — Next.js + LangGraph

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app), integrated with a **LangGraph-based AI agent** for HR automation.

> ✨ The LangGraph agent code is located in the `/agent` directory at the root level.

---

## 🚀 System Requirements

Ensure the following dependencies are installed before setup:

- **Node.js** v20.19.1 (LTS)
- **npm** (comes with Node.js)
- **Python** 3.10.9
- **yarn** 1.22.22
- **pip** 25.1.1

---

## ⚙️ Environment Setup

1. **Create environment file**  
   Copy the provided draft configuration:

   ```bash
   cp .draft.env.local .env.local
   ```

2. **Configure the following variables in `.env.local`:**

   ```env
   NEXT_PUBLIC_BASE_URL=https://68121212e81df7060eb6f6d2.mockapi.io
   # This is required by the React boilerplate for API service (even if not directly used)

   OPENAI_API_KEY=your-openai-api-key
   GPT_MODEL_NAME=gpt-4.1-nano     # Use this version for best results
   LANGCHAIN_API_KEY=your-langchain-api-key
   LANGCHAIN_PROJECT=your-project-name
   LANGCHAIN_TRACING_V2=true
   ```

---

## 📦 Install Dependencies

### 1. Install JavaScript packages
From the project root:

```bash
yarn install
```

> Or use `npm install`, `pnpm install`, or `bun install` depending on your setup.

### 2. Install Python packages
Create a virtual environment if desired:

```bash
# Create virtual env (optional)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

This project requires **two servers**:
Please run them in two separate terminals.

| Component        | Command                        | Port   |
|------------------|--------------------------------|--------|
| Next.js Frontend | `yarn dev` or `npm run dev`    | 3000   |
| Flask HR Agent   | `python -m agent.HRAgent`      | 5000   |

### Run the frontend server:

```bash
yarn dev
# or
npm run dev
```

### Run the LangGraph agent:

```bash
python -m agent.HRAgent
```

---

## 🌐 Access the App

Visit: [http://localhost:3000](http://localhost:3000)

Login using any valid credentials that meet the format requirements:

```txt
Email:    test@gmail.com
Password: test12345678
```

> ⚠️ If ports `3000` or `5000` are already in use, please update your configuration accordingly.

---

## 📸 Screenshots

### 🧭 LangGraph State Graph  
![State Graph](https://github.com/user-attachments/assets/11912983-7d42-446b-a023-27924a174aec)

### 📊 LangChain Tracing Dashboard  
![LangSmith Tracing](https://github.com/user-attachments/assets/8a60400b-052e-495c-83f7-89eca009fa39)

### 💬 Running Chat Application  
![Chat App](https://github.com/user-attachments/assets/de174b72-0850-4af6-aa26-cbeaa67a026c)

---

## 📌 Notes

- Stick with `gpt-4.1-nano` for model consistency.
- `NEXT_PUBLIC_BASE_URL` is a required placeholder for the API structure of the boilerplate—your app may fail to fetch otherwise.
- Tracing with LangChain is optional but enabled via `LANGCHAIN_TRACING_V2`.

---

## 📄 License

Licensed under the [MIT License](LICENSE).