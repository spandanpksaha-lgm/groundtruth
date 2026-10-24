"""The document sets GroundTruth knows about. Each corpus is fully separate:
its own documents, index, refusal threshold, test questions and results.
Nothing is ever retrieved across corpora.

To add another corpus, add an entry here, put its files in `data_dir`, and
write its questions in eval/questions_<key>_{test,dev}.jsonl.
"""
from __future__ import annotations

CORPORA: dict[str, dict] = {
    "uc": {
        "name": "UC Student Handbook",
        "heading": "answers about the University of the Cumberlands Student Handbook, with receipts",
        "data_dir": "data/uc/text",
        "base_url": "",
        # The original PDF, so citations can open the exact page.
        "source_file": "data/uc/Student_Handbook_2026_2027.pdf",
        "about": "Corpus: University of the Cumberlands Student Handbook 2026-2027 (PDF, 81 pages; "
                 "pages 30-33 are drug-schedule tables and are not indexed).",
        "disclaimer": "Not official university advice — confirm with the Registrar or Student Affairs.",
        "placeholder": "Ask about withdrawal, financial aid, parking, housing, conduct…",
        "examples": [
            "How long after the last day of class do I have to file a formal written appeal of a grade?",
            "I got seriously ill and can't attend classes. How soon do I have to submit a medical withdrawal request?",
            "Am I allowed to ride my skateboard on campus sidewalks?",
            "How much is undergraduate tuition per credit hour?",
            "How much is the late fee if I miss my tuition payment due date?",
        ],
        "examples_label": "Try these (the skateboard question hits a contradiction in the handbook; "
                          "the last two are trick questions)",
        # "What can I ask?" on the main tab. Every answerable question here is from the verified
        # test/dev sets (its answer and page were checked against the handbook).
        "categories": [
            ("Grades, appeals and withdrawal", [
                "How long after the last day of class do I have to file a formal written appeal of a grade?",
                "I got seriously ill and can't attend classes. How soon do I have to submit a medical withdrawal request?",
                "I'm an on-campus student. During the fall semester, how many in-seat hours must I take before I can "
                "register for 6 hours of online classes?",
            ]),
            ("Financial aid and work-study", [
                "How many credit hours do I need each semester to count as a full-time undergraduate for financial aid?",
                "What cumulative GPA must an undergraduate keep to maintain Satisfactory Academic Progress for financial aid?",
                "What is the pay rate for most Federal Work-Study jobs on campus?",
            ]),
            ("Parking, vehicles and getting around", [
                "How many parking tickets in one semester get my car towed, and how much does it cost to get it back?",
                "How long do I have to dispute a campus parking ticket?",
                "What's the speed limit for electric scooters on the campus shared-use paths?",
            ]),
            ("Housing and residence life", [
                "Can I use an air fryer in my dorm room? What about a coffee maker?",
                "How many nights per semester can an overnight guest stay with me in the residence hall?",
                "When do quiet hours start and end on a Friday night in the residence halls?",
            ]),
            ("Campus services and graduation", [
                "How many hours of community service do main campus undergraduates need to graduate?",
                "How long will my university email account stay active after I graduate?",
                "How far in advance do I need to request the free airport shuttle?",
            ]),
            ("Tricky ones: watch it say I don't know, or flag a conflict", [
                "How much is undergraduate tuition per credit hour?",
                "How much is the late fee if I miss my tuition payment due date?",
                "Am I allowed to ride my skateboard on campus sidewalks?",
            ]),
        ],
        # 2x2 Lab demo questions, picked from the team benchmark runs (Oct 24) because the four
        # setups disagree on them. The model is not perfectly repeatable, so rehearse first.
        "lab_examples": [
            "I missed every assignment in the first two weeks of an online course. Does the handbook say "
            "my enrollment is definitely cancelled?",
            "Who receives the towing retrieval payment, and what does the amount start at?",
            "For Spring 2027 graduate and undergraduate online second bi-term, when does it end, and is that the same day as the main-campus undergraduate second bi-term?",
            "I left for a designated break without returning my key, and my room failed inspection for "
            "two separately cited items. What minimum total fines could apply?",
            "How much is undergraduate tuition per credit hour?",
        ],
    },
    "tts": {
        "name": "TTS Travel & Leave",
        "heading": "answers about TTS travel & leave policy, with receipts",
        "data_dir": "data/raw",
        "base_url": "https://handbook.tts.gsa.gov",
        "source_file": None,
        "about": "Corpus: GSA TTS Handbook, Travel & Leave section (public domain, June 2026 snapshot).",
        "disclaimer": "Not official HR advice — confirm with HR.",
        "placeholder": "Ask about leave, travel, reimbursement…",
        "examples": [
            "How much annual leave can I carry over into the next leave year?",
            "What happens to my unused travel comp time if I leave GSA?",
            "If I take all 12 weeks of paid parental leave, how much FMLA do I have left?",
            "What is the daily meals and incidental expenses rate in Boston?",
            "Can I fly business class on an international flight longer than 14 hours?",
        ],
        "examples_label": "Try these (the last two are trick questions)",
    },
}


def corpus_spec(key: str) -> dict:
    if key not in CORPORA:
        raise SystemExit(f"Unknown corpus '{key}'. Choose one of: {', '.join(CORPORA)}")
    return CORPORA[key]
