"""A tiny fake OpenAI-compatible server for offline tests of the LLM code path.
It is NOT a model: it echoes the best-matching sentence from the passages with
a citation, and occasionally misbehaves on purpose (no citation) so the
safety gates get exercised.

    python tests/fake_llm_server.py 8765 &
    LLM_PROVIDER=openai LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_MODEL=fake python eval/run_eval.py
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def answer(user: str) -> str:
    q = user.rsplit("Question:", 1)[-1].lower()
    qw = set(re.findall(r"[a-z0-9]+", q)) - {"the", "a", "i", "my", "do", "is", "what", "how", "can", "of", "to", "in"}
    if "judge" in user.lower():
        return "CORRECT"
    best = (0, None, None)
    for m in re.finditer(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", user, re.S):
        for s in re.split(r"(?<=[.!?])\s+|\n", m.group(2)):
            ov = len(qw & set(re.findall(r"[a-z0-9]+", s.lower())))
            if ov > best[0]:
                best = (ov, m.group(1), s.strip())
    if best[0] < 2:
        return "I don't know — the documents I have don't cover this."
    if "pet" in q:                       # deliberately uncited -> gate 2 should block it
        return "Yes, pet fees are reimbursable as an incidental expense."
    return f"{best[2]} [{best[1]}]"


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        sys_txt = msgs[0]["content"]
        text = answer(("judge " if "grade" in sys_txt or "check citations" in sys_txt else "") + msgs[-1]["content"])
        if "check citations" in sys_txt:
            text = "SUPPORTED"
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
