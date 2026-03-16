"""
Gradio UI — Radiology AI with Human-in-the-Loop
"""
import logging
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv("/home/moez/projects/radiology-ai/.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pipeline.preprocessor import preprocess
from agents.orchestrator import run_pipeline, resume_pipeline


def process_scan(dicom_file, modality: str):
    if dicom_file is None:
        return None, "", "", "", "", "⚠️ Please upload a DICOM file.", gr.update(visible=False)

    try:
        result = preprocess(dicom_path=dicom_file.name, output_dir="data/processed")
        png_path      = result["png_path"]
        anonymized_id = result["anonymized_id"]

        state, thread_id = run_pipeline(
            png_path=png_path,
            anonymized_id=anonymized_id,
            modality=modality,
            hil=True,
        )

        report_text = ""
        qa_summary  = ""
        status      = state.get("status", "unknown")
        show_hil    = False

        if state.get("report"):
            report_text = state["report"].report_text

        if state.get("validation"):
            v = state["validation"]
            qa_summary = f"{'✅ Passed' if v.passed else '⚠️ Failed'} — Score: {v.score}\n"
            if v.issues:
                qa_summary += f"Issues: {', '.join(v.issues)}\n"
            if v.warnings:
                qa_summary += f"Warnings: {', '.join(v.warnings)}\n"
            if v.requires_human_review:
                qa_summary += "🔴 Flagged for human review"
                show_hil = True

        if status == "complete":
            status_msg = "✅ Pipeline complete — report ready"
        elif status in ("interrupted", "qa_failed", "validated"):
            if show_hil:
                status_msg = "⏸️ Awaiting radiologist review"
            else:
                status_msg = "✅ Pipeline complete — report ready"
        else:
            status_msg = f"ℹ️ Status: {status}"

        scan_info = (
            f"Modality: {modality} | "
            f"Urgency: {state['report'].urgency_level if state.get('report') else 'N/A'} | "
            f"ID: {anonymized_id}"
        )

        return (
            png_path,
            report_text,
            qa_summary,
            scan_info,
            thread_id,
            status_msg,
            gr.update(visible=show_hil),
        )

    except Exception as e:
        logger.error("UI error: %s", e, exc_info=True)
        return None, "", "", "", "", f"❌ Error: {str(e)}", gr.update(visible=False)


def approve_report(report_text: str, thread_id: str):
    if not report_text.strip():
        return "⚠️ No report to approve.", gr.update(visible=True)
    try:
        final_state = resume_pipeline(
            thread_id=thread_id,
            approved_report=report_text,
            approved=True,
        )
        return (
            f"✅ Report approved and saved. ID: {thread_id} | Status: {final_state.get('status')}",
            gr.update(visible=False),
        )
    except Exception as e:
        logger.error("Approve error: %s", e, exc_info=True)
        return f"❌ Error approving: {str(e)}", gr.update(visible=True)


def reject_report(thread_id: str):
    try:
        resume_pipeline(thread_id=thread_id, approved_report="", approved=False)
        return f"🔴 Report rejected. ID: {thread_id}", gr.update(visible=False)
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(visible=True)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Radiology AI") as demo:

    gr.Markdown("""
    # 🏥 Radiology AI — Multi-Agent Report Generation
    Upload a DICOM scan to generate an AI-assisted radiology report.
    All patient data is anonymized on upload (GDPR compliant).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Scan")
            dicom_input    = gr.File(label="DICOM file (.dcm)", file_types=[".dcm"])
            modality_input = gr.Dropdown(
                choices=["CR", "MR", "CT", "DX", "US"],
                value="CR", label="Modality",
            )
            analyze_btn = gr.Button("🔍 Analyze Scan", variant="primary")
            gr.Markdown("### Scan Preview")
            scan_image  = gr.Image(label="Preprocessed scan (512x512)", type="filepath")
            scan_info   = gr.Textbox(label="Scan info", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### AI Generated Report")
            status_box   = gr.Textbox(label="Status", interactive=False)
            report_box   = gr.Textbox(
                label="Radiology report (editable before approving)",
                lines=16, interactive=True,
            )
            qa_box       = gr.Textbox(label="QA validation", lines=5, interactive=False)
            thread_state = gr.State("")

            with gr.Group(visible=False) as hil_panel:
                gr.Markdown("### 👨‍⚕️ Radiologist Review Required")
                gr.Markdown("You can edit the report above before approving.")
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve Report", variant="primary")
                    reject_btn  = gr.Button("🔴 Reject", variant="stop")
                action_output = gr.Textbox(label="Action result", interactive=False)

    analyze_btn.click(
        fn=process_scan,
        inputs=[dicom_input, modality_input],
        outputs=[scan_image, report_box, qa_box, scan_info, thread_state, status_box, hil_panel],
    )
    approve_btn.click(
        fn=approve_report,
        inputs=[report_box, thread_state],
        outputs=[action_output, hil_panel],
    )
    reject_btn.click(
        fn=reject_report,
        inputs=[thread_state],
        outputs=[action_output, hil_panel],
    )

    gr.Markdown("---\n*AI-generated reports must be reviewed by a qualified radiologist.*")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
