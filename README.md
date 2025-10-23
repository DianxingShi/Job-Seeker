
#  Job-Seeker
> "You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations - Release Code

---

## 🚀 Overview

This repository contains the complete codebase for an empirical study on the capabilities of Large Language Models (LLMs) in the context of corporate hiring evaluations.

The project is a sophisticated, dual-LLM framework designed to simulate a complete personality assessment pipeline:
1.  **🤖 The Candidate:** An LLM is prompted to act as a job candidate (specifically, a Software Engineer) applying for a position at a top tech company. It is given strategic instructions to answer a 233-question personality quiz in a way that maximizes its chances of being hired.
2.  **🧑‍⚖️ The HR Evaluator:** A second LLM is prompted to act as a seasoned HR expert. This LLM receives the "candidate's" answers and performs a detailed qualitative analysis, assessing personality traits, job fit, and providing a final hiring recommendation.

This framework allows for robust testing and evaluation of how different LLMs perform in both "gaming" the assessment and "evaluating" the results, all benchmarked against a quantitative "ideal" answer key.

## 📦 Project Components

This repository is structured into several key components that work together to execute the experiment.

| Component | Files | Purpose |
| :--- | :--- | :--- |
| 🌐 **Standalone Quiz** | [`quiz.html`](./quiz.html) | A standalone, bilingual (EN/CN) web-based personality quiz used for data collection from humans or LLMs. |
| 🔬 **Core Framework** | [`job-seeker/`](./job-seeker/) | The main Flask application, containing the two simulation backends (`candidate.py`, `hr.py`) and their web UIs. |
| 🧠 **Prompting** | [`prompt_template.txt`](./prompt_template.txt) | The "brain" of the experiment, containing the detailed system prompts that define the roles and strategies for the LLMs. |
| 📊 **Analysis** | [`job-seeker/calculate.py`](./job-seeker/calculate.py) | A Python script for quantitative analysis, comparing an LLM's answers to a "ground truth" baseline using RMSE and Pearson Correlation. |
| 🗃️ **Quiz Data** | `quiz_EN.txt`, `quiz_CN.txt`, `quiz_with_standard_answers_CN.txt` | The raw questions in English and Chinese, plus the "ideal" answer key for quantitative scoring. |

---

## 💡 How It Works: The Experimental Workflow

The entire study is designed to be run in a clear, multi-step process:

### Step 1: Generate Candidate Answers (Simulation)

1.  **Run the Candidate Server:**
    ```bash
    python job-seeker/candidate.py
    ```
   
2.  **Open the Web UI:** Access the "Job-hunting Assistant" at `http://localhost:5000`.
3.  **Configure the "Candidate":**
    * Select an LLM model (e.g., `gpt-4o`, `gemini2.5-pro`).
    * Copy the `CANDIDATE_SYSTEM_PROMPT_EN` (or `_CN`) from `prompt_template.txt` and paste it as the "task description." This prompt instructs the LLM on *how* to answer strategically.
4.  **Run the Quiz:**
    * Copy all 233 questions from `quiz_EN.txt` (or `quiz_CN.txt`).
    * Upload this list as a `.txt` file using the upload button.
5.  **Save Results:**
    * The "candidate" LLM will generate a numerical answer (1-9) for each question.
    * Use the "Save Conversation" 💾 button. This saves the complete list of questions and the LLM's answers to a new `.json` file in the `candidate_conversations/` directory.

### Step 2: Evaluate Candidate Answers (Simulation)

1.  **Run the HR Server:**
    ```bash
    python job-seeker/hr.py
    ```
   
2.  **Open the Web UI:** Access the "HR Assistant" at `http://localhost:5001`.
3.  **Configure the "HR Expert":**
    * Select an LLM model to act as the evaluator.
    * Copy the `HR_SYSTEM_PROMPT_EN` (or `_CN`) from `prompt_template.txt` and paste it as the "role description." This prompt instructs the LLM on *how* to evaluate the answers for a "Software Engineer" role.
4.  **Run the Evaluation:**
    * Upload the `.json` file of the candidate's answers that was generated in **Step 1**.
5.  **Get Analysis:**
    * The "HR" LLM will provide a full qualitative report, including:
        * 3-5 prominent personality traits.
        * A job fit assessment (strengths and risks).
        * A final hiring recommendation: **Strongly Recommend**, **Recommend with Reservations**, or **Not Recommended**.
    * The analysis can be saved to the `hr_analyses/` directory.

### Step 3: Quantitative Analysis (Scoring)

1.  **Configure the Script:**
    * Open `job-seeker/calculate.py`.
    * Set `json_filename` to the path of the candidate's answer file (from **Step 1**).
    * Set `answers_filename` to the path of the baseline file (e.g., `quiz_with_standard_answers_CN.txt`).
2.  **Run the Analysis:**
    ```bash
    python job-seeker/calculate.py
    ```
   
3.  **Get Scores:**
    * The script outputs the **RMSE (Root Mean Square Error)** and **Pearson Correlation** between the LLM's answers and the "ideal" answers.
    * It also provides a detailed breakdown of scores for key "positive" (green-flag) and "negative" (red-flag) questions.

---

## 🔑 Key File Descriptions

### Core Application
* **`job-seeker/candidate.py`**
    * A Flask backend that serves the "Candidate" simulation. It manages API calls to various LLM providers, maintains the conversation history, and uses the `CANDIDATE_SYSTEM_PROMPT` to guide the LLM's behavior.
* **`job-seeker/hr.py`**
    * A Flask backend that serves the "HR Evaluator" simulation. It receives a `.json` file of answers, formats it into a prompt, and uses the `HR_SYSTEM_PROMPT` to have an LLM generate a detailed hiring report.
* **`job-seeker/templates/candidate.html`**
    * The frontend for `candidate.py`. A modern, chat-based web interface for selecting a model, setting the task, and interacting with the "candidate" LLM.
* **`job-seeker/templates/hr.html`**
    * The frontend for `hr.py`. A web interface for selecting a model, setting the HR role, and uploading a candidate's answer file for analysis.

### Experimental Logic & Analysis
* **`prompt_template.txt`**
    * **This is the most critical file for the experiment's design.**
    * **`CANDIDATE_SYSTEM_PROMPT` (EN/CN):** Instructs the LLM to be a Software Engineer. Crucially, it provides explicit strategies to pass the test, such as "Emphasize 'Team Cooperation'", "Demonstrate 'Rigorous and Meticulous' Work Style", and identifies "Trap' Questions to Be Cautious About."
    * **`HR_SYSTEM_PROMPT` (EN/CN):** Instructs the LLM to be an HR expert at a top tech company. It defines the exact 3-part structure required for the evaluation (Personality Analysis, Job Fit Assessment, Conclusion).
* **`job-seeker/calculate.py`**
    * The quantitative analysis script. It calculates the statistical similarity between a candidate's answers and the "standard" answers, allowing for objective performance measurement of different "candidate" LLMs.

### Data & Assets
* **`quiz.html`**
    * A single, standalone HTML file that functions as a complete personality quiz app. It features:
        * 216 questions.
        * A 1-9 point Likert scale ("Very Strongly Disagree" to "Very Strongly Agree").
        * Bilingual support (Chinese & English).
        * Export functionality to download the user's answers as `.txt` or `.json`.
* **`quiz_with_standard_answers_CN.txt`**
    * The "ground truth" or "ideal" answer key. Each line contains one of the 233 questions followed by the "correct" score (1-9) for a desirable candidate.
* **`quiz_CN.txt`** / **`quiz_EN.txt`**
    * Simple text files containing the 216 questions, used for feeding into the "candidate" LLM.

---

## 🔌 Model & API Integration

The backends (`candidate.py` and `hr.py`) are built to be highly flexible, supporting multiple major LLM providers.

### Supported API Types
* **`azure`**: Azure OpenAI Service
* **`openai_compatible`**: Any endpoint compatible with the OpenAI API (e.g., DeepSeek, Dashscope's Qwen)
* **`azure_inference`**: Azure AI Inference SDK
* **`gemini`**: Google Gemini API

### Supported Models (from config)
* **GPT**: `gpt-4o`, `gpt-5-chat`
* **Qwen**: `qwen1.5-32b-chat`, `qwen2.5-32b-instruct`, `qwen3-32b`
* **DeepSeek**: `deepseek-r1`, `deepseek-v3.1`, `deepseek-v3.2-exp`, `deepseek-v3.1-terminus`
* **Gemini**: `gemini2.5-flash`, `gemini2.5-pro`
* **Llama**: `llama-3.3-70b-instruct`, `llama4-maverick-17b-128e-instruct-fp8`

---

## 🛠️ Setup and Usage

### 1. Prerequisites
* Python 3.x
* Flask
* Required Python SDKs (based on the models you want to use):
    * `openai` (for Azure and OpenAI-compatible)
    * `azure-ai-inference`
    * `google-generativeai`
* `numpy`
* `scipy`

Install dependencies:
```bash
pip install Flask openai azure-ai-inference google-generativeai numpy scipy
