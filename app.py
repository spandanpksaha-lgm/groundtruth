"""GroundTruth demo UI.

Tabs:
  1. UC Student Handbook  - the assistant. A settings panel switches between the
                            four systems of the 2x2 grid (chunking x embedding).
  2. UC 2x2 Lab           - one question, four systems side by side.
  3. TTS Travel & Leave   - the same pipeline on a second, unrelated document set.

The document sets never mix: each has its own index, threshold and examples.

    python app.py            # then open http://127.0.0.1:7860
    python app.py --share    # temporary public link (handy on demo-day Wi-Fi)
    python app.py --only uc  # load just the UC tabs
"""
import argparse
import html
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import gradio as gr

from groundtruth.config import ROOT, get_config
from groundtruth.corpora import CORPORA, corpus_spec
from groundtruth.grid import cell_config, cell_label, cells, grid_models
from groundtruth.llm import DailyLimitError, make_llm
from groundtruth.rag import GroundTruth

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
GRID_CORPUS = "uc"

REASONS = {
    "low_retrieval": "No passage in the documents was relevant enough, so the model was never asked.",
    "model_declined": "Relevant-looking passages were found, but the model judged they don't answer the question.",
    "no_citation": "The model produced an answer without citing a passage, so it was blocked.",
    "llm_error": "The language model could not be reached (rate limit?). Try again in a minute.",
}
SHORT_REASONS = {
    "low_retrieval": "nothing relevant enough was retrieved",
    "model_declined": "the model found the passages don't answer it",
    "no_citation": "answer had no citation, so it was blocked",
    "llm_error": "language model unreachable (rate limit?)",
}

CSS = """
.gt-card {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; margin-bottom:8px;}
.gt-card.cited {border-left:4px solid #2a78d6;}
.gt-meta {font-size:12px; color:var(--body-text-color-subdued); margin-bottom:4px;}
.gt-text {font-size:13px; white-space:pre-wrap;}
.gt-badge {display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; margin-left:6px;
           background:#2a78d6; color:#fff;}
.gt-page {display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; margin-left:6px;
          border:1px solid var(--border-color-primary);}
.gt-warn {color:#b3541e; font-size:12px;}
.gt-cell {border:1px solid var(--border-color-primary); border-radius:12px; padding:12px 14px; min-height:220px;}
.gt-cell h4 {margin:0 0 6px 0; font-size:14px;}
.gt-verdict {display:inline-block; font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px;
             margin-bottom:8px;}
.gt-verdict.ans {background:#e5f4e4; color:#1d5e1b;}
.gt-verdict.idk {background:#fdf0e6; color:#8a3f12;}
.gt-verdict.err {background:#fbebeb; color:#8c1d1d;}
.gt-answer {font-size:13.5px; line-height:1.45; margin-bottom:8px;}
.gt-src {font-size:12px; color:var(--body-text-color-subdued); margin:2px 0;}
.gt-ex {gap:6px !important;}
.gt-ex button {white-space: normal !important; text-align: left !important; justify-content: flex-start !important;
    height: auto !important; min-height: 0 !important; padding: 6px 10px !important; font-weight: 400 !important;
    line-height: 1.35 !important;}
.gt-ex-label {font-size:13px; font-weight:600; color:#4f46e5;}
.gt-help {font-size:13.5px; line-height:1.5;}
.gt-help table {font-size:13px;}
.gt-axis {font-size:12px; font-weight:600; color:var(--body-text-color-subdued); text-align:center;}
"""
EMPTY = "<i>Ask a question to see the evidence.</i>"
CELL_EMPTY = "<div class='gt-cell'><i>Waiting for a question.</i></div>"


def source_link(spec, chunk):
    """Web pages link to their URL; PDF-derived chunks open the PDF at the page."""
    if chunk.url.startswith("http"):
        return chunk.url
    if spec.get("source_file") and (ROOT / spec["source_file"]).exists():
        path = quote(str(ROOT / spec["source_file"]))
        return f"/gradio_api/file={path}" + (f"#page={chunk.page}" if chunk.page else "")
    return ""


def render_sources(spec, gt, a):
    if not a.hits:
        return "<i>No passages retrieved.</i>"
    cited = set(a.cited_ranks)
    weak = a.refusal_reason == "low_retrieval"
    out = [f"<div class='gt-meta'>{html.escape(describe(gt))} · top retrieval score {a.top_score:.3f} · "
           f"refusal threshold {gt.threshold:.3f} · {a.latency_s}s"
           + (" · <b>closest passages, not relevant enough to answer</b>" if weak else "") + "</div>"]
    for i, h in enumerate(a.hits, 1):
        c = h.chunk
        badge = "<span class='gt-badge'>cited</span>" if i in cited else ""
        page = f"<span class='gt-page'>p.{c.page}</span>" if c.page else ""
        link = source_link(spec, c)
        title = (f"<a href='{html.escape(link)}' target='_blank'>{html.escape(c.title)}</a>"
                 if link else html.escape(c.title))
        text = html.escape(c.text if len(c.text) < 1200 else c.text[:1200] + " …")
        out.append(
            f"<div class='gt-card{' cited' if i in cited else ''}'>"
            f"<div class='gt-meta'><b>[{i}]</b> {title} — {html.escape(c.section)}{page} · "
            f"score {h.score:.3f}{badge}</div><div class='gt-text'>{text}</div></div>")
    return "\n".join(out)


def answer_body(a):
    if a.refused:
        return f"{a.text}\n\n<span class='gt-warn'>Why: {REASONS.get(a.refusal_reason, a.refusal_reason)}</span>"
    body = a.text
    weak = [c for c in a.checks if not c.supported]
    if weak:
        body += ("\n\n<span class='gt-warn'>⚠ Support check: "
                 f"{len(weak)} sentence(s) use wording not found in the cited passage. "
                 "Verify against the source.</span>")
    pages = sorted({h.chunk.page for h in a.cited_hits if h.chunk.page})
    if pages:
        body += f"\n\n<span class='gt-meta'>Source page(s): {', '.join(map(str, pages))}.</span>"
    body += "\n\n<span class='gt-meta'>Check the cited passages on the right before acting on this.</span>"
    return body


def describe(gt):
    cfg = gt.cfg
    return f"{cfg.chunker} chunks ({cfg.chunk_size}w) · {cfg.embedder_label} embeddings"


def chat_component():
    kw = dict(height=460, label="Conversation", render_markdown=True, sanitize_html=False)
    if GRADIO_MAJOR < 6:
        kw["type"] = "messages"
    return gr.Chatbot(**kw)


def examples(qs, box, label):
    """Clickable example questions that show their full text (gr.Examples cuts long ones off with '…').
    Clicking one copies it into the question box."""
    gr.Markdown(f"<span class='gt-ex-label'>{html.escape(label)}</span>")
    with gr.Column(elem_classes="gt-ex"):
        for q in qs:
            gr.Button(q, size="sm", variant="secondary").click(lambda q=q: q, None, box)


def make_tab(key, systems, default):
    """One assistant tab. `systems` maps a cell key to a GroundTruth; with more
    than one, a settings panel picks which one answers."""
    spec = corpus_spec(key)
    chunk_opts = list(dict.fromkeys(k[0] for k in systems))
    model_opts = list(dict.fromkeys(k[1] for k in systems))

    def respond(question, history, chunk_choice, model_choice):
        history = history or []
        if not question.strip():
            return history, gr.update(), ""
        gt = systems.get((chunk_choice, model_choice)) or systems[default]
        try:
            a = gt.ask(question)
        except DailyLimitError:
            return history + [{"role": "user", "content": question},
                              {"role": "assistant", "content": "The language model's daily quota is used up. "
                               "Switch LLM_API_KEY in .env to another key and restart the app."}], gr.update(), ""
        body = answer_body(a)
        if len(systems) > 1:
            body += f"\n\n<span class='gt-meta'>Answered by: {html.escape(describe(gt))}.</span>"
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": body}]
        return history, render_sources(spec, gt, a), ""

    with gr.Tab(spec["name"]):
        gr.Markdown(f"### GroundTruth — {spec['heading']}\n"
                    "Every answer cites the passages it used. If the document doesn't say, "
                    "it says *I don't know*.")
        with gr.Row():
            with gr.Column(scale=5):
                chat = chat_component()
                box = gr.Textbox(placeholder=spec["placeholder"], show_label=False)
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                if spec.get("categories"):
                    gr.Markdown("#### What can I ask?\nThe handbook covers academics, financial aid, parking, "
                                "housing, conduct and campus services. Click a question to try it. It does "
                                "*not* cover tuition prices or anything outside the handbook; for those it "
                                "should say *I don't know*.")
                    for title, qs in spec["categories"]:
                        examples(qs, box, title)
                elif spec["examples"]:
                    examples(spec["examples"], box, spec["examples_label"])
            with gr.Column(scale=4):
                if len(systems) > 1:
                    with gr.Accordion("Settings: chunking and embedding model", open=False):
                        chunk_r = gr.Radio([(c.capitalize(), c) for c in chunk_opts], value=default[0],
                                           label="Chunking strategy")
                        model_r = gr.Radio([(cell_label(m), m) for m in model_opts], value=default[1],
                                           label="Embedding model")
                        gr.Markdown("<span class='gt-meta'>Each combination has its own index and its own "
                                    "calibrated “I don't know” threshold. The 2×2 Lab tab runs all four at "
                                    "once.</span>")
                else:
                    chunk_r = gr.State(default[0])
                    model_r = gr.State(default[1])
                gr.Markdown("#### Retrieved passages")
                sources = gr.HTML(EMPTY)
        gt = systems[default]
        gr.Markdown(f"<span class='gt-meta'>{spec['about']} Model: {gt.cfg.llm_provider} "
                    f"{getattr(gt.llm, 'model', '')} · default: {html.escape(describe(gt))}. "
                    f"{spec['disclaimer']}</span>")
        ins = [box, chat, chunk_r, model_r]
        send.click(respond, ins, [chat, sources, box])
        box.submit(respond, ins, [chat, sources, box])
        clear.click(lambda: ([], EMPTY, ""), None, [chat, sources, box])


def render_cell(spec, label, gt, a):
    if a is None:
        return f"<div class='gt-cell'><h4>{html.escape(label)}</h4><i>Not available: index not built.</i></div>"
    if a.refused:
        cls = "err" if a.refusal_reason == "llm_error" else "idk"
        verdict = f"<span class='gt-verdict {cls}'>I don't know · {SHORT_REASONS.get(a.refusal_reason, '')}</span>"
        body = ""
    else:
        verdict = f"<span class='gt-verdict ans'>Answered · cites {len(a.cited_ranks)} passage(s)</span>"
        text = html.escape(a.text)
        text = re.sub(r"\[(\d+)\]", r"<b>[\1]</b>", text).replace("\n", "<br>")
        body = f"<div class='gt-answer'>{text}</div>"
        weak = [c for c in a.checks if not c.supported]
        if weak:
            body += (f"<div class='gt-warn'>⚠ {len(weak)} sentence(s) use wording not found in the "
                     "cited passage.</div>")
    gate = "passed" if a.top_score >= gt.threshold else "below"
    meta = (f"<div class='gt-src'>Top score {a.top_score:.3f} vs threshold {gt.threshold:.3f} "
            f"({gate}) · {a.latency_s}s</div>")
    cited = set(a.cited_ranks)
    rows = []
    for i, h in enumerate(a.hits[:3], 1):
        c = h.chunk
        link = source_link(spec, c)
        page = f"p.{c.page}" if c.page else ""
        page_html = f"<a href='{html.escape(link)}' target='_blank'>{page}</a>" if link and page else page
        mark = "<b>cited</b> · " if i in cited else ""
        rows.append(f"<div class='gt-src'>[{i}] {mark}{html.escape(c.section[:70])} {page_html} · "
                    f"{h.score:.3f}</div>")
    return (f"<div class='gt-cell'><h4>{html.escape(label)}</h4>{verdict}{body}{meta}"
            f"<div class='gt-src' style='margin-top:6px'><b>Top 3 passages</b></div>{''.join(rows)}</div>")


def benchmark_table(key, m1, m2):
    """2x2 table of the team benchmark, read from eval/results if the runs exist."""
    import csv
    from groundtruth.grid import run_tag
    good = {"correct", "correct_refusal"}
    vals = {}
    for ch, model, _ in cells(key):
        path = ROOT / "eval" / "results" / run_tag(key, "hard", ch, model) / "per_question.csv"
        if path.exists():
            rows = list(csv.DictReader(open(path)))
            vals[(ch, cell_label(model))] = f"{sum(r['outcome'] in good for r in rows)}/{len(rows)}"
    if not vals:
        return "*Run `bash run_grid_uc.sh` to fill in this table.*"
    get = lambda ch, m: vals.get((ch, m), "not run")  # noqa: E731
    return (f"| | {m1} | {m2} |\n|---|---|---|\n"
            f"| **Section chunks** | {get('section', m1)} | {get('section', m2)} |\n"
            f"| **Fixed chunks** | {get('fixed', m1)} | {get('fixed', m2)} |")


def lab_explainer(key, m1, m2):
    return f"""
---
### What's going on here

**One question, four complete systems.** Every box uses the same handbook, the same language model and the
same prompt. Only the *retrieval* step differs:

- **Rows: how the handbook is cut into passages.** *Section chunks* follow the handbook's own headings, and
  each passage starts with its heading path (e.g. "Student Life > Parking"). *Fixed chunks* are 220-word
  windows cut regardless of headings, so a rule can be split from the heading that says what it's about.
- **Columns: which embedding model turns text into vectors.** Questions and passages become points in the
  same vector space (the *latent space*); retrieval returns the passages closest to the question. {m1} and
  {m2} are two different models trained to put similar meanings close together, and they disagree on what
  "close" means for some questions.

**Reading a box**

- **Answered · cites N passage(s)**: the answer must cite the numbered passages it used; an answer with no
  citation is blocked.
- **I don't know**: either *nothing relevant enough was retrieved* (the top score fell below that system's
  threshold, so the model was never asked), or *the model found the passages don't answer it*.
- **Top score vs threshold**: cosine similarity of the best passage, compared with a cut-off calibrated
  separately for each system on held-out questions. {m2} scores everything higher than {m1}, so only
  compare a score with its own threshold.
- **Top 3 passages**: what the model was shown, with page links into the PDF. **cited** marks the ones the
  answer relied on.

**Why the four disagree.** The model can only answer from what retrieval hands it, p(answer | question,
passages). Change the passages and you change the answer. When a fixed window cuts a rule away from its
heading, the right passage may not be found, or may be found without the context the model needs.

**What the demo questions showed in our benchmark run** (the model's answers can vary slightly between runs):

1. *Missed assignments in an online course*: section chunks answered; both fixed-chunk systems said
   "I don't know".
2. *Towing payment*: fixed + {m1} refused (nothing scored above its threshold); fixed + {m2} found it.
   A better embedding can rescue a badly cut passage.
3. *Spring 2027 online second bi-term end date*: same section chunks, different embedding, different
   answer. Section + {m1} got it right; section + {m2} gave the wrong date; fixed + {m1} refused.
4. *Key and inspection fines*: the two fines sit in different sections; section + {m1} missed one passage.
5. *Tuition per credit hour*: not in the handbook. All four should say "I don't know".

**Team benchmark: 40 questions written by the team, correct out of 40**

{benchmark_table(key, m1, m2)}

Correct = right answer with a supporting citation, or a correct refusal. All four refused all 8
not-in-the-handbook questions. A 1-2 question gap is within run-to-run noise; section vs fixed chunking held
across both models. Full per-question results: `eval/results/UC_benchmark_results.xlsx`.

<span class='gt-meta'>Each click makes up to four language-model calls; on the free Groq tier, wait a few
seconds between questions.</span>
"""


def make_lab_tab(key, grid):
    """2x2 Lab: one question through all four systems."""
    spec = corpus_spec(key)
    labels = [lab for _, _, lab in cells(key)]
    order = [(ch, m) for ch, m, _ in cells(key)]

    def run_all(question):
        if not question.strip():
            return [gr.update()] * 5
        def one(k):
            gt = grid.get(k)
            if not gt:
                return None
            try:
                return gt.ask(question)
            except DailyLimitError:
                return "quota"
        with ThreadPoolExecutor(max_workers=4) as ex:
            answers = list(ex.map(one, order))
        if "quota" in answers:
            return (["**The language model's daily quota is used up.** Switch LLM_API_KEY in .env to another "
                     "key and restart the app."] + [CELL_EMPTY] * 4)
        htmls = [render_cell(spec, lab, grid.get(k), a) for k, lab, a in zip(order, labels, answers)]
        ok = [a for a in answers if a is not None]
        n_ans = sum(not a.refused for a in ok)
        pages = {k: sorted({h.chunk.page for h in a.cited_hits if h.chunk.page})
                 for k, a in zip(order, answers) if a is not None and not a.refused}
        agree = len({tuple(p) for p in pages.values()}) <= 1
        summary = (f"**{n_ans} of {len(ok)} systems answered**, {len(ok) - n_ans} said *I don't know*."
                   + (" The answering systems cited different pages." if pages and not agree else ""))
        return [summary] + htmls

    with gr.Tab("UC 2×2 Lab"):
        m1, m2 = (cell_label(m) for m in grid_models())
        gr.Markdown("### 2×2 Lab — same question, four retrieval setups\n"
                    f"Rows: **section-aware** vs **fixed-size** chunks (220 words). "
                    f"Columns: **{m1}** vs **{m2}** embeddings. Same language model, same prompt, same "
                    "document; only retrieval changes. Each system's similarity score is compared with its "
                    "own threshold (the models score on different scales).")
        with gr.Row():
            box = gr.Textbox(placeholder="Ask one question; all four systems answer it",
                             show_label=False, scale=6)
            go = gr.Button("Run all four", variant="primary", scale=1)
        summary = gr.Markdown("")
        outs = []
        with gr.Row():
            gr.HTML(f"<div class='gt-axis'>{html.escape(m1)}</div>")
            gr.HTML(f"<div class='gt-axis'>{html.escape(m2)}</div>")
        for r in range(2):
            with gr.Row(equal_height=True):
                for c in range(2):
                    outs.append(gr.HTML(CELL_EMPTY))
        lab_ex = spec.get("lab_examples") or spec["examples"]
        if lab_ex:
            examples(lab_ex, box, "Demo questions (the setups disagreed on the first four in our benchmark; "
                     "the last is a trick question all four should refuse). Click one, then Run all four.")
        gr.Markdown(lab_explainer(key, m1, m2), elem_classes="gt-help")
        go.click(run_all, [box], [summary] + outs)
        box.submit(run_all, [box], [summary] + outs)


def load_grid(llm):
    grid = {}
    for ch, model, label in cells(GRID_CORPUS):
        try:
            grid[(ch, model)] = GroundTruth(cell_config(GRID_CORPUS, ch, model), llm=llm)
            print(f"  loaded {label}: threshold {grid[(ch, model)].threshold:.3f}")
        except Exception as e:  # missing model download, no network, ...
            print(f"  skipped {label}: {e}")
    return grid


def build(only=None):
    keys = [only] if only else list(CORPORA)
    style = dict(css=CSS, theme=gr.themes.Soft())
    llm = make_llm(get_config())
    grid = load_grid(llm) if GRID_CORPUS in keys else {}
    with gr.Blocks(title="GroundTruth", **({} if GRADIO_MAJOR >= 6 else style)) as demo:
        gr.Markdown("## GroundTruth — a RAG assistant that shows its receipts")
        for k in keys:
            if k == GRID_CORPUS and grid:
                default = next(iter(grid))
                make_tab(k, grid, default)
                make_lab_tab(k, grid)
            else:
                make_tab(k, {("default", "default"): GroundTruth(get_config(corpus=k), llm=llm)},
                         ("default", "default"))
    allowed = [str(ROOT / s["source_file"]) for s in CORPORA.values() if s.get("source_file")]
    return demo, style, allowed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--only", choices=list(CORPORA), help="load a single document set")
    a = p.parse_args()
    demo, style, allowed = build(a.only)
    demo.launch(share=a.share, server_port=a.port, allowed_paths=allowed,
                **(style if GRADIO_MAJOR >= 6 else {}))
