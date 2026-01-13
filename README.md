# Orchestrated Blog Generator (LangGraph + Streamlit)

This project is a example of an **agentic, orchestrator–worker style LLM application** built with **LangGraph**, **LangChain**, and **Streamlit**.  
It takes a blog topic as input and generates a full technical blog post by coordinating multiple structured LLM calls through a graph-based workflow.

The goal of this project is not just to generate text, but to demonstrate how to **design, control, and compose LLM behaviors** using explicit orchestration, typed state, and deterministic flow control.

---
This is the Snapshot of the project

<img width="906" height="408" alt="Screenshot 2026-01-13 144202" src="https://github.com/user-attachments/assets/24ef2998-cf11-4b86-88d4-ee9da86423d7" />

---
## Overview

The application implements an **orchestrator–worker–synthesizer pipeline**:

1. **Orchestrator**
   - Generates the blog title and TL;DR.
   - Produces a structured outline (5–8 sections), each with a name and description.

2. **Workers**
   - One worker is spawned per section.
   - Each worker generates a concrete, instructional markdown section based on the outline.

3. **Synthesizer**
   - Collects all generated sections.
   - Assembles the final blog post with a title, TL;DR, and section separators.

The entire pipeline is expressed as a **LangGraph state machine**, which makes the control flow explicit, inspectable, and extensible.

---

## Architecture

<img width="145" height="217" alt="Screenshot 2026-01-13 145234" src="https://github.com/user-attachments/assets/f8735753-f8cb-4167-9fa9-691faa829040" />

---

## Key Concepts Demonstrated

- Graph-based control flow using LangGraph (`StateGraph`, `START`, `END`)
- Strongly-typed LLM outputs using Pydantic models (`Sections`, `BlogMeta`)
- Explicit state passing between nodes via typed dictionaries
- Fan-out / fan-in orchestration using conditional edges and `Send`
- LLM role specialization (planner vs. section writer vs. synthesizer)
- Human-friendly UI built with Streamlit
- Deterministic assembly of final content from independent LLM calls

---

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain + LangGraph
- Pydantic
- Groq LLM API (`openai/gpt-oss-120b`)
- python-dotenv

---

## How It Works

1. The user enters a blog topic.
2. The orchestrator:
   - Creates a title and TL;DR.
   - Generates a structured outline.
3. Each outline section is sent to a worker node that writes the content.
4. The synthesizer joins everything into a single markdown document.
5. The result is displayed and can be downloaded as a `.md` file.

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit langchain langgraph langchain-groq pydantic python-dotenv




