"""GroundTruth demo UI.

    python app.py            # then open http://127.0.0.1:7860
    python app.py --share    # temporary public link (handy on demo-day Wi-Fi)
"""
import argparse
import html

import gradio as gr

GRADIO_MAJOR = int(gr.__version__.split(".")[0])

from groundtruth.config import get_config
from groundtruth.rag import GroundTruth

cfg = get_config()
gt = GroundTruth(cfg)

EXAMPLES = [
    "How much annual leave can I carry over into the next leave year?",
    "What happens to my unused travel comp time if I leave GSA?",
    "If I take all 12 weeks of paid parental leave, how much FMLA do I have left?",
    "What is the daily meals and incidental expenses rate in Boston?",   # trick question
    "Can I fly business class on an international flight longer than 14 hours?",  # trick
]

REASONS = {
    "low_retrieval": "No passage in the documents was relevant enough, so the model was never asked.",
    "model_declined": "Relevant-looking passages were found, but the model judged they don't answer the question.",
    "no_citation": "The model produced an answer without citing a passage, so it was blocked.",
    "llm_error": "The language model could not be reached.",
}

CSS = """
.gt-card {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; margin-bottom:8px;}
.gt-card.cited {border-left:4px solid #2a78d6;}
.gt-meta {font-size:12px; color:var(--body-text-color-subdued); margin-bottom:4px;}
.gt-text {font-size:13px; white-space:pre-wrap;}
.gt-badge {display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; margin-left:6px;
           background:#2a78d6; color:#fff;}
.gt-refuse {border-left:4px solid #ec835a; padding:8px 12px; border-radius:8px; background:var(--block-background-fill);}
.gt-warn {color:#b3541e; font-size:12px;}
"""


def render_sources(a):
    if not a.hits:
        return "<i>No passages retrieved.</i>"
    cited = set(a.cited_ranks)
    out = [f"<div class='gt-meta'>Top retrieval score {a.top_score:.3f} · refusal threshold "
           f"{gt.threshold:.3f} · {a.latency_s}s</div>"]
    for i, h in enumerate(a.hits, 1):
        c = h.chunk
        badge = "<span class='gt-badge'>cited</span>" if i in cited else ""
        text = html.escape(c.text if len(c.text) < 1200 else c.text[:1200] + " …")
        out.append(
            f"<div class='gt-card{' cited' if i in cited else ''}'>"
            f"<div class='gt-meta'><b>[{i}]</b> <a href='{html.escape(c.url)}' target='_blank'>"
            f"{html.escape(c.title)}</a> — {html.escape(c.section)} · score {h.score:.3f}{badge}</div>"
            f"<div class='gt-text'>{text}</div></div>")
    return "\n".join(out)


def respond(question, history):
    history = history or []
    if not question.strip():
        return history, "", ""
    a = gt.ask(question)
    if a.refused:
        body = f"{a.text}\n\n<span class='gt-warn'>Why: {REASONS.get(a.refusal_reason, a.refusal_reason)}</span>"
    else:
        body = a.text
        weak = [c for c in a.checks if not c.supported]
        if weak:
            body += ("\n\n<span class='gt-warn'>⚠ Support check: "
                     f"{len(weak)} sentence(s) use wording not found in the cited passage. "
                     "Verify against the source.</span>")
        body += "\n\n<span class='gt-meta'>Check the cited passages on the right before acting on this.</span>"
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": body}]
    return history, render_sources(a), ""


STYLE = dict(css=CSS, theme=gr.themes.Soft())
with gr.Blocks(title="GroundTruth", **({} if GRADIO_MAJOR >= 6 else STYLE)) as demo:
    gr.Markdown("## GroundTruth — answers about TTS travel & leave policy, with receipts\n"
                "Every answer cites the passages it used. If the handbook doesn't say, it says *I don't know*.")
    with gr.Row():
        with gr.Column(scale=5):
            chat_kwargs = dict(height=460, label="Conversation", render_markdown=True, sanitize_html=False)
            if GRADIO_MAJOR < 6:
                chat_kwargs["type"] = "messages"   # Gradio 5 needs this; 6 uses messages by default
            chat = gr.Chatbot(**chat_kwargs)
            box = gr.Textbox(placeholder="Ask about leave, travel, reimbursement…", show_label=False)
            with gr.Row():
                send = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(EXAMPLES, inputs=box, label="Try these (the last two are trick questions)")
        with gr.Column(scale=4):
            gr.Markdown("### Retrieved passages")
            sources = gr.HTML("<i>Ask a question to see the evidence.</i>")
    gr.Markdown(f"<span class='gt-meta'>Corpus: TTS Handbook, Travel & Leave (public domain). "
                f"Model: {cfg.llm_provider} {getattr(gt.llm, 'model', '')} · embeddings: {cfg.embedder} · "
                f"chunking: {cfg.chunker} {cfg.chunk_size}w. Not official HR advice — confirm with HR.</span>")

    send.click(respond, [box, chat], [chat, sources, box])
    box.submit(respond, [box, chat], [chat, sources, box])
    clear.click(lambda: ([], "<i>Ask a question to see the evidence.</i>", ""), None, [chat, sources, box])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    demo.launch(share=a.share, server_port=a.port, **(STYLE if GRADIO_MAJOR >= 6 else {}))
