# Demo script, slide outline, ethics & Q&A prep

## Weekend plan (who does what)

| When | Product | Model/Prompt | Integration | Evaluation | Ethics & Risk |
|---|---|---|---|---|---|
| Fri night | Confirm the story; read 3–4 handbook pages | Get an LLM key working; run `run_all.sh` once | Get `app.py` running on 2 laptops | Read every test question; add 2–3 of your own *before* seeing outputs | Draft harms list (below) |
| Sat midday (CP1) | Draft slides 1–3 | Tune the prompt only if the failure log shows a pattern | Record a rough demo | First real numbers | Pick 2 harms to demo |
| Sat evening (CP2) | Full deck | Try a second model or embedder | Backup video | `make_report.py`; screenshot failures | Mitigations slide |
| Sat 10 PM | **Code freeze** | | | | |

## Slide outline (10–12 slides)

1. **The problem.** "Annual leave over 240 hours is forfeited. Ask a chatbot and get it wrong, and you lose a week of vacation."
2. **Who it hurts.** Employees acting on policy answers alone, especially new hires and people on leave.
3. **What we built.** One screenshot: answer + highlighted cited passage + source link.
4. **Architecture.** The mermaid diagram from the README; call out the two refusal gates.
5. **Live demo** (script below).
6. **Evaluation design.** 30 questions, 7 deliberately unanswerable, written first; the categories table.
7. **Results.** `eval/results/metrics.png` with real numbers.
8. **Failure analysis.** `outcomes.png` plus the 2 most interesting rows from the failure log, each labeled as a retrieval or generation problem.
9. **Stretch goal.** `chunking_comparison.png`: what chunk size and structure did to retrieval.
10. **Ethics & risk.** Specific harms → specific mitigations (below).
11. **What we'd build next.**
12. *(optional)* Most Honest Failure.

## Demo script (rehearse twice; use exactly these)

1. **Easy hit.** "How much annual leave can I carry over into the next leave year?" → 240 hours, cited. Click the source link to show it's real.
2. **The confusable pair.** "What happens to my unused travel comp time if I leave GSA?" → *forfeited*. Then say: "Regular comp time is the opposite: it's paid out. A model that blurs these costs someone money."
3. **The trick question.** "What is the daily meals and incidental expenses rate in Boston?" The handbook has an M&IE example for DC ($69), and "Boston" appears on another page. Retrieval finds related text, so this one tests the model, not just the search. Expected: "I don't know."
4. *(If time)* "Can I fly business class on an international flight longer than 14 hours?" Many models "know" a rule here from training. GroundTruth should decline because the handbook doesn't say.

**Before demo day:** run all four on the exact laptop, LLM and Wi-Fi you'll use. If #3 or #4 *doesn't* refuse, keep it and make it your Most Honest Failure slide rather than hiding it. Record the backup video Saturday night.

## Ethics & risk: specific harms → concrete mitigations

| Harm | How it happens here | Mitigation (built or proposed) |
|---|---|---|
| **Confident wrong answer** | Model blends near-identical facts (comp time vs. travel comp time) or uses training knowledge (mileage rates) | Two refusal gates; answers must cite; support check flags unsupported wording; eval measures this directly |
| **Citation that looks right but isn't** | The answer cites a real passage that doesn't actually contain the claim. Citations *increase* trust, so this is worse than no citation | `citation_faithfulness` and `sentence_support_rate` metrics; cited passages shown beside the answer so users can check |
| **Stale policy** | Handbook pages are years old (07/13/2021 COVID travel guidance; salary cap "as of 2024") | Show source links; *proposed:* show each page's last-updated date and warn when it's over a year old |
| **Over-reliance** | Employees act on the answer without checking, especially for high-stakes leave decisions | UI says "check the cited passages," footer says "not official HR advice"; *proposed:* for high-stakes topics (FMLA, PPL, LWOP), always add "confirm with your HR specialist" |
| **Refusing too much** | A system tuned to never be wrong refuses real questions, and people stop using it | `false_refusal_rate` is reported next to refusal accuracy; threshold calibrated on dev to avoid refusing answerable questions |
| **Scope creep** | Users ask about their personal situation ("Am I eligible?") | The system answers from policy text only and never sees personal data; *proposed:* detect personal-eligibility questions and point to HR |
| **Prompt injection via documents** | If someone could edit the corpus, text in a page could instruct the model | The corpus is a fixed, reviewed snapshot; passages are framed as data in the prompt. Worth one line: "we'd need provenance checks before indexing live pages" |

## What we'd build next

- Show each page's last-modified date; down-rank or flag stale pages
- Hybrid retrieval (TF-IDF + dense) and a reranker
- Answer-level confidence calibrated on a larger dev set
- Expand to the full 240-page handbook; a regression test set that runs on every handbook update

## Q&A prep

- **"How do you know the citations are real?"** Three ways: the evidence-string check (automatic), the sentence support rate (automatic), and the optional LLM judge. Plus the user sees the passage.
- **"Isn't keyword-matching correctness too strict or too loose?"** Yes, it can be. That's why we added the LLM judge and spot-checked failures by hand. Have one example ready where they disagree.
- **"Why not just a bigger model?"** Bigger models know *more* outside facts, which makes the Boston and business-class traps more tempting, not less. The eval will show whether that's true.
- **"Did you tune on the test set?"** No. Questions were written first; the threshold was calibrated on a separate dev set.
- **"Why this corpus?"** Public domain, real policy, specific numbers that are easy to grade, and wrong answers with real costs.
