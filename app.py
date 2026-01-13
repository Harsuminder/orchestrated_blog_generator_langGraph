
import os
from dotenv import load_dotenv
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

import streamlit as st
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END


# =========================
# App + Backend
# =========================

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class Section(BaseModel):
    name: str = Field(description="Name for this section of the blog post")
    description: str = Field(description="Brief overview of what this blog section will cover")


class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the blog post")


class BlogMeta(BaseModel):
    title: str = Field(description="A compelling, clear blog post title")
    tldr: str = Field(description="A short TL;DR summary (2-4 sentences)")


class State(TypedDict):
    topic: str
    title: str
    tldr: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_blog: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


@st.cache_resource(show_spinner=False)
def build_workflow():
    llm = ChatGroq(model="openai/gpt-oss-120b")

    outline_planner = llm.with_structured_output(Sections)
    meta_planner = llm.with_structured_output(BlogMeta)

    def orchestrator(state: State):
        meta = meta_planner.invoke(
            [
                SystemMessage(content="You are a technical blog editor. Produce a strong title and a concise TL;DR."),
                HumanMessage(content=f"Blog topic: {state['topic']}"),
            ]
        )

        outline = outline_planner.invoke(
            [
                SystemMessage(
                    content=(
                        "Create a detailed outline for a technical tutorial blog post. "
                        "Return 5–8 sections. Each section must be practical and logically ordered."
                    )
                ),
                HumanMessage(content=f"Blog topic: {state['topic']}"),
            ]
        )

        return {
            "title": meta.title,
            "tldr": meta.tldr,
            "sections": outline.sections,
        }

    def llm_call(state: WorkerState):
        section = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a blog section for a technical tutorial. "
                        "No preamble. Use markdown. Be concrete and instructional. "
                        "Assume the reader is a developer. Do not add SEO metadata."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Section name: {state['section'].name}\n"
                        f"Section description: {state['section'].description}\n\n"
                        "Write this section now."
                    )
                ),
            ]
        )
        return {"completed_sections": [section.content]}

    def assign_workers(state: State):
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    def synthesizer(state: State):
        body = "\n\n---\n\n".join(state["completed_sections"])
        final_blog = f"# {state['title']}\n\n**TL;DR:** {state['tldr']}\n\n---\n\n{body}"
        return {"final_blog": final_blog}

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("llm_call", llm_call)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
    builder.add_edge("llm_call", "synthesizer")
    builder.add_edge("synthesizer", END)

    workflow = builder.compile()
    return workflow


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Orchestrated Blog Generator (LangGraph)", layout="wide")
st.title("Orchestrated Blog Generator (LangGraph)")
st.caption("An agentic blog generation pipeline implementing an orchestrator–worker pattern with LangGraph and LLMs.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your environment or a .env file, then restart the app.")
    st.stop()

workflow = build_workflow()

with st.form("blog_form"):
    topic = st.text_input(
        "Blog topic",
        value=st.session_state.get("topic", "Building a Blog Generator with an Orchestrator-Worker Pattern in LangGraph"),
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        run = st.form_submit_button("Generate blog")
    with col2:
        clear = st.form_submit_button("Clear output")

if clear:
    for k in ["topic", "final_blog", "title", "tldr"]:
        st.session_state.pop(k, None)
    st.rerun()

if run:
    st.session_state["topic"] = topic
    with st.spinner("Generating blog (orchestrator → workers → synthesize)..."):
        result_state = workflow.invoke({"topic": topic})
        st.session_state["final_blog"] = result_state["final_blog"]
        # (Optional) keep these for display/debug
        st.session_state["title"] = result_state.get("title", "")
        st.session_state["tldr"] = result_state.get("tldr", "")

final_blog = st.session_state.get("final_blog")

if final_blog:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Blog Output (Markdown)")
        st.markdown(final_blog)

    with right:
        st.subheader("Export")
        default_filename = "blog_post.md"
        st.download_button(
            label="Download .md",
            data=final_blog.encode("utf-8"),
            file_name=default_filename,
            mime="text/markdown",
        )

        st.subheader("Raw Markdown")
        st.text_area("Copy/paste", value=final_blog, height=380)

else:
    st.info("Enter a topic and click **Generate blog**.")
