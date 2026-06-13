# AI Study Buddy Team

A 3-agent system, built with **LangGraph**, that helps you study any topic:

```
Researcher -> Quizzer -> (you answer) -> Grader -> Simplifier (if score is low)
```

Built as a homework project for the **Generative AI Applications with
LangChain** workshop at MBM College ("Multi-agent focus" assignment).

## How it works

The agents share one **state object** that flows through the graph:

```
topic -> research -> quiz -> answers -> feedback -> simplified
```

1. **Researcher agent** — given a topic, writes a concise summary of 5-8 key
   points using Groq (`qwen/qwen3-32b`).
2. **Quizzer agent** — reads the research notes and writes 3 quiz questions
   (without answers).
3. **You answer** — the quiz is printed to the terminal; type your answers,
   then type `DONE`.
4. **Grader agent** — compares your answers to the research notes, gives
   per-question feedback, and an overall score `X/10`.
5. **Simplifier agent (bonus)** — if your score is below 6/10, a fourth agent
   automatically re-explains the topic in simpler terms with analogies,
   focusing on what you got wrong. If you scored 6+, this step is skipped.

This routing decision (Simplifier vs. skip) is made with LangGraph's
**conditional edges** — the graph itself decides which path to take based on
the Grader's output.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key

Copy `.env.example` to `.env` and add your key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

## Usage

```bash
python main.py
```

Example session:

```
What topic do you want to study today? Photosynthesis

[Researcher] Looking up key points...
[Quizzer] Writing quiz questions...

--- Quiz ---
1. What are the two main stages of photosynthesis?
2. Which organelle is responsible for photosynthesis?
3. What are the inputs and outputs of the overall reaction?

Type your answer to each question. When finished, type DONE on its own line.
> light reactions and the Calvin cycle
> chloroplast
> inputs: CO2, water, light. outputs: glucose, oxygen
> DONE

[Grader] Checking your answers...

--- Feedback ---
...
SCORE: 8/10
```

If the score were below 6, a "Simplified explanation" section would also be
printed automatically.

## Project structure

```
study-buddy-team/
├── main.py           # Main application (graph definition + agents)
├── requirements.txt  # Python dependencies
├── .env.example      # Template for API key
└── README.md
```

## Tools used

- **LangGraph** — defines the multi-agent graph and routing logic
- **LangChain** — LLM wrapper
- **Groq** (`qwen/qwen3-32b`) — LLM used by all four agents

## Possible extensions

- Let the Quizzer ask follow-up questions based on weak areas
- Save research notes + feedback to a file for later review
- Add a difficulty setting (easy/medium/hard quiz questions)
