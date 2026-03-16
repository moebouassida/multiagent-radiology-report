"""
LangGraph Orchestrator with Human-in-the-Loop
----------------------------------------------
The pipeline can pause at the human_review node and wait
for a radiologist to approve/edit the report before continuing.

Two modes:
  - auto: runs fully autonomously (no HIL)
  - hil:  pauses when QA fails or human review is flagged,
          waits for human input, then resumes
"""
import logging
import os
import sqlite3
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from agents.image_analysis import ImageFindings, run as analyze
from agents.clinical_context import ClinicalContext, run as get_context
from agents.report_drafting import RadiologyReport, run as draft_report
from agents.qa_validation import ValidationResult, run as validate

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DB_PATH = "data/checkpoints.db"   # SQLite stores paused pipeline states


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    png_path: str
    anonymized_id: str
    modality: str
    image_findings: ImageFindings | None
    clinical_context: ClinicalContext | None
    report: RadiologyReport | None
    validation: ValidationResult | None
    retry_count: int
    error: str | None
    status: str
    # HIL fields
    human_approved: bool          # True once radiologist approves
    final_report_text: str        # may be edited by radiologist


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════
def node_image_analysis(state: PipelineState) -> PipelineState:
    """Runs vision model on the scan, extracts findings."""
    logger.info("[node] image_analysis | anon_id=%s", state["anonymized_id"])
    try:
        findings = analyze(
            png_path=state["png_path"],
            anonymized_id=state["anonymized_id"],
            modality=state["modality"],
        )
        return {**state, "image_findings": findings, "status": "image_analyzed"}
    except Exception as e:
        logger.error("[node] image_analysis failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_clinical_context(state: PipelineState) -> PipelineState:
    """Fetches relevant medical knowledge for the findings."""
    logger.info("[node] clinical_context | anon_id=%s", state["anonymized_id"])
    try:
        context = get_context(state["image_findings"])
        return {**state, "clinical_context": context, "status": "context_retrieved"}
    except Exception as e:
        logger.error("[node] clinical_context failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_report_drafting(state: PipelineState) -> PipelineState:
    """Generates the full structured radiology report."""
    logger.info(
        "[node] report_drafting | anon_id=%s | attempt=%d",
        state["anonymized_id"],
        state["retry_count"] + 1,
    )
    try:
        report = draft_report(state["image_findings"], state["clinical_context"])
        return {**state, "report": report, "status": "report_drafted"}
    except Exception as e:
        logger.error("[node] report_drafting failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_qa_validation(state: PipelineState) -> PipelineState:
    """Validates the report for completeness, consistency, urgency."""
    logger.info("[node] qa_validation | anon_id=%s", state["anonymized_id"])
    try:
        result = validate(state["report"], state["image_findings"])
        return {
            **state,
            "validation": result,
            "retry_count": state["retry_count"] + (0 if result.passed else 1),
            "status": "validated" if result.passed else "qa_failed",
        }
    except Exception as e:
        logger.error("[node] qa_validation failed: %s", e)
        return {**state, "error": str(e), "status": "failed"}


def node_human_review(state: PipelineState) -> PipelineState:
    """
    HIL NODE — this is where the magic happens.

    interrupt() pauses the entire graph execution here.
    LangGraph saves the full state to SQLite.
    The graph stays paused until someone calls graph.invoke()
    again with the same thread_id and a Command(resume=...).

    When resumed, `human_input` contains whatever the radiologist
    passed back — their edited report text or an approval signal.

    The node then updates the state with the human-approved report
    and sets status = "human_approved" so the graph can finish.
    """
    logger.info("[node] human_review — PAUSING for radiologist | anon_id=%s", state["anonymized_id"])

    # interrupt() pauses here and sends this dict to the UI
    # the UI displays it to the radiologist and waits for input
    human_input = interrupt({
        "message": "Please review and approve the radiology report.",
        "report": state["report"].report_text if state["report"] else "",
        "qa_score": state["validation"].score if state["validation"] else 0,
        "qa_issues": state["validation"].issues if state["validation"] else [],
        "anonymized_id": state["anonymized_id"],
    })

    # execution resumes here after radiologist responds
    # human_input is whatever was passed to Command(resume=...)
    approved_text = human_input.get("approved_report", state["report"].report_text)
    approved = human_input.get("approved", False)

    logger.info(
        "[node] human_review — RESUMED | approved=%s | anon_id=%s",
        approved,
        state["anonymized_id"],
    )

    return {
        **state,
        "human_approved": approved,
        "final_report_text": approved_text,
        "status": "human_approved" if approved else "human_rejected",
    }


def node_finalize(state: PipelineState) -> PipelineState:
    """
    Final node — report is approved (by QA or human).
    In phase 5 this will save to PostgreSQL and S3.
    For now just logs and sets final status.
    """
    final_text = state.get("final_report_text") or (
        state["report"].report_text if state["report"] else ""
    )
    logger.info(
        "[node] finalize | anon_id=%s | human_approved=%s",
        state["anonymized_id"],
        state.get("human_approved", False),
    )
    return {
        **state,
        "final_report_text": final_text,
        "status": "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_after_analysis(state: PipelineState) -> str:
    return "end" if state["status"] == "failed" else "continue"


def route_after_qa(state: PipelineState) -> str:
    """
    After QA:
      - failed pipeline  -> end
      - QA passed + no human review needed -> finalize directly
      - QA passed + human review flagged   -> human_review node
      - QA failed + retries left           -> retry report drafting
      - QA failed + max retries            -> force human review
    """
    if state["status"] == "failed":
        return "end"

    validation = state.get("validation")

    if validation and validation.passed:
        if validation.requires_human_review:
            return "human_review"    # passed but flagged — radiologist should see it
        return "finalize"            # clean pass — save directly

    if state["retry_count"] >= MAX_RETRIES:
        logger.warning("Max retries — forcing human review")
        return "human_review"        # give up retrying, human takes over

    return "retry"                   # retry report drafting


def route_after_human(state: PipelineState) -> str:
    """After human review: approved -> finalize, rejected -> end."""
    if state.get("human_approved"):
        return "finalize"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_graph(checkpointer=None):
    """
    Builds and compiles the LangGraph pipeline.

    checkpointer: if provided, enables HIL (state persistence between
                  interrupt and resume). Pass None for fully auto mode.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("image_analysis",  node_image_analysis)
    graph.add_node("clinical_context", node_clinical_context)
    graph.add_node("report_drafting",  node_report_drafting)
    graph.add_node("qa_validation",    node_qa_validation)
    graph.add_node("human_review",     node_human_review)
    graph.add_node("finalize",         node_finalize)

    graph.set_entry_point("image_analysis")

    graph.add_conditional_edges(
        "image_analysis",
        route_after_analysis,
        {"continue": "clinical_context", "end": END},
    )
    graph.add_edge("clinical_context", "report_drafting")
    graph.add_edge("report_drafting",  "qa_validation")
    graph.add_conditional_edges(
        "qa_validation",
        route_after_qa,
        {
            "finalize":     "finalize",
            "human_review": "human_review",
            "retry":        "report_drafting",
            "end":          END,
        },
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"finalize": "finalize", "end": END},
    )
    graph.add_edge("finalize", END)

    return graph.compile(
        checkpointer=checkpointer,
        # interrupt_before tells LangGraph to pause BEFORE entering human_review
        # so the state is saved before any HIL logic runs
        interrupt_before=["human_review"] if checkpointer else [],
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    png_path: str,
    anonymized_id: str,
    modality: str = "CR",
    hil: bool = False,
    thread_id: str | None = None,
) -> tuple[PipelineState, str | None]:
    """
    Run the pipeline.

    Args:
        hil:       if True, enables human-in-the-loop with SQLite checkpointing
        thread_id: unique ID for this pipeline run (needed for HIL resume)

    Returns:
        (state, thread_id)
        If state["status"] == "interrupted" -> waiting for human input
        If state["status"] == "complete"    -> done
    """
    load_dotenv("/home/moez/projects/radiology-ai/.env")
    os.makedirs("data", exist_ok=True)

    thread_id = thread_id or anonymized_id

    if hil:
        # SQLite checkpointer saves state between interrupt and resume
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
    else:
        graph = build_graph()
        config = {}

    initial_state: PipelineState = {
        "png_path":          png_path,
        "anonymized_id":     anonymized_id,
        "modality":          modality,
        "image_findings":    None,
        "clinical_context":  None,
        "report":            None,
        "validation":        None,
        "retry_count":       0,
        "error":             None,
        "status":            "started",
        "human_approved":    False,
        "final_report_text": "",
    }

    logger.info("Starting pipeline | anon_id=%s | hil=%s", anonymized_id, hil)
    final_state = graph.invoke(initial_state, config)
    logger.info("Pipeline status: %s", final_state.get("status"))

    return final_state, thread_id


def resume_pipeline(
    thread_id: str,
    approved_report: str,
    approved: bool = True,
) -> PipelineState:
    """
    Resume a paused pipeline after human review.

    Args:
        thread_id:        same thread_id used in run_pipeline()
        approved_report:  the (possibly edited) report text from the radiologist
        approved:         True = approve and finalize, False = reject
    """
    load_dotenv("/home/moez/projects/radiology-ai/.env")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Resuming pipeline | thread_id=%s | approved=%s", thread_id, approved)

    # Command(resume=...) sends data back to the interrupt() call in node_human_review
    # the value here is what `human_input` receives inside that node
    final_state = graph.invoke(
        Command(resume={
            "approved_report": approved_report,
            "approved": approved,
        }),
        config,
    )

    logger.info("Pipeline resumed | status=%s", final_state.get("status"))
    return final_state
