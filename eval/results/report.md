# GroundTruth evaluation report

Test set: 30 hand-written questions (23 answerable, 7 deliberately unanswerable), written before looking at model output. Definitions are in `groundtruth/evaluation.py`.

| Metric | `test-section220-tfidf-nollm` | `test-fixed220-tfidf-nollm` |
|---|---|---|
| answer_correctness | 0.696 | 0.739 |
| citation_faithfulness | 0.773 | 0.857 |
| refusal_accuracy | 0.429 | 0.429 |
| false_refusal_rate | 0.043 | 0.087 |
| retrieval_hit@5 | 0.913 | 0.913 |
| sentence_support_rate | 1.0 | 1.0 |
| mrr | 0.775 | 0.79 |
| harm_weighted_error | 0.333 | 0.302 |
| avg_latency_s | 0.0 | 0.0 |

![metrics](metrics.png)

![outcomes](outcomes.png)

## Chunking comparison (retrieval only)

| Embedder | Chunker | Size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC | Missed |
|---|---|---|---|---|---|---|---|---|---|
| tfidf | fixed | 120 | 251 | 70% | 78% | 83% | 0.74 | 0.92 | t15,t16,t18,t20 |
| tfidf | fixed | 220 | 118 | 70% | 91% | 91% | 0.79 | 0.91 | t18,t20 |
| tfidf | fixed | 400 | 64 | 83% | 91% | 96% | 0.87 | 0.92 | t20 |
| tfidf | section | 120 | 254 | 65% | 83% | 91% | 0.76 | 0.91 | t16,t20 |
| tfidf | section | 220 | 178 | 65% | 91% | 91% | 0.78 | 0.88 | t16,t20 |
| tfidf | section | 400 | 148 | 70% | 91% | 96% | 0.81 | 0.86 | t20 |

![chunking](chunking_comparison.png)

## Failure log

### Failures — `test-section220-tfidf-nollm`

| ID | Outcome | Question | What it said | Top score |
|---|---|---|---|---|
| t02 | wrong_answer | I've been a federal employee for 5 years. How much annual leave do I earn each pay period? | You will earn 56 more annual leave hours (14 pay periods x 4 hours per pay period). [2] You earn 4 annual leave hours per pay period and there are 14 pay period… | 0.367 |
| t09 | wrong_answer | If I'm approved for administrative leave to vote on election day, what is the maximum amount of time I can take? | In rare circumstances, authorized agency officials may approve limited amounts of administrative leave on election day when there is no reasonable opportunity f… | 0.3113 |
| t11 | wrong_answer | When does regular comp time expire, and what happens to it if I leave GSA? | If you leave GSA, any unused travel comp time will be forfeited. [1] If you leave GSA, your comp time will be paid out. [2] | 0.2641 |
| t14 | wrong_answer | How long do I need to have worked for the federal government to be eligible for FMLA? | Annual leave accrues based on how long you’ve worked for the federal government:. [2] PPL is how the federal government implements FMLA for parental leave. [1] | 0.1764 |
| t16 | retrieval_miss | After paid parental leave, how long do I have to agree to return to work? | TTS has paid parental leave (PPL) for the birth or placement (adoption or foster care) of a child. [1] Refer to the Paid Parental Leave page for more informatio… | 0.2881 |
| t17 | correct_bad_citation | How much bereavement leave can I use per year for a civilian family member, and how much funeral leave do I get for an immediate relative's combat-related death? | You can use up to 13 days (104 hours) per year of your sick leave for bereavement, funeral planning, and/or attending the funeral of a family member. [2] You ca… | 0.4455 |
| t18 | false_refusal | Above what dollar amount do I need a receipt for a travel expense? | I don't know — the documents I have don't cover this. | 0.134 |
| t20 | correct_bad_citation | How soon after returning from a trip should I submit my travel voucher? | While use of the GSA travel card is required for all travel expenses when practicable, using the GSA travel card to pay for M&IE requires some additional admini… | 0.1399 |
| t23 | wrong_answer | What counts as a 'large team gathering' that needs the event request process instead of an approval email? | You must formally request your authorizing official's approval* via email and forward it to tts-travel@gsa.gov, unless you are traveling to a training, conferen… | 0.2886 |
| u01 | hallucinated | What is the maximum per diem lodging rate for a hotel in Chicago? | As long as all lodging and meals expenses are within the maximum per diem rate for the location, and direct rental car costs (not including gas/tolls) are not m… | 0.2165 |
| u02 | hallucinated | What is the daily meals and incidental expenses rate in Boston? | The allowance for Meals and Incidental Expenses (M&IE) is a flat rate given to a traveler regardless of what they actually spent. [1] How should I claim meals a… | 0.3117 |
| u05 | hallucinated | How many paid federal holidays do GSA employees get each year? | If you are a new federal employee, check out the recommendations for those who have less than 1 year of federal service. - You need to be approved for PPL befor… | 0.1248 |
| u07 | hallucinated | Can I expense a pet boarding fee while I'm on official travel? | Include language that specifies which days will be at your own expense, and acknowledge that "I understand all other travel expenses including lodging and meals… | 0.1427 |

### Failures — `test-fixed220-tfidf-nollm`

| ID | Outcome | Question | What it said | Top score |
|---|---|---|---|---|
| t09 | wrong_answer | If I'm approved for administrative leave to vote on election day, what is the maximum amount of time I can take? | In rare circumstances, authorized agency officials may approve limited amounts of administrative leave on election day when there is no reasonable opportunity f… | 0.2929 |
| t11 | wrong_answer | When does regular comp time expire, and what happens to it if I leave GSA? | If you leave GSA, any unused travel comp time will be forfeited. [1] If you leave GSA, your comp time will be paid out. [2] | 0.1811 |
| t16 | wrong_answer | After paid parental leave, how long do I have to agree to return to work? | Submitting paid parental leave requests in HR Links After your Workforce Relations HR Specialist reviews your supporting documentation, you will see a new leave… | 0.1768 |
| t18 | retrieval_miss | Above what dollar amount do I need a receipt for a travel expense? | I don't know — the documents I have don't cover this. | 0.1534 |
| t20 | retrieval_miss | How soon after returning from a trip should I submit my travel voucher? | I don't know — the documents I have don't cover this. | 0.1091 |
| t23 | wrong_answer | What counts as a 'large team gathering' that needs the event request process instead of an approval email? | How to secure authorizing official approval You must formally request your authorizing official's approval* via email and forward it to tts-travel@gsa.gov, unle… | 0.2349 |
| u01 | hallucinated | What is the maximum per diem lodging rate for a hotel in Chicago? | An increase in overall trip cost that is within 120% of the originally authorized amount.* As long as all lodging and meals expenses are within the maximum per … | 0.197 |
| u02 | hallucinated | What is the daily meals and incidental expenses rate in Boston? | Questions about entering specific expenses How should I claim meals and incidental expenses? - The allowance for Meals and Incidental Expenses (M&IE) is a flat … | 0.2067 |
| u03 | hallucinated | What is the reimbursement rate per mile for using my personal car on official travel? | Brief overview of vouchers You may be reimbursed for your expenses from official travel, or any other approved expense paid for via personal funds (such as reim… | 0.1223 |
| u07 | hallucinated | Can I expense a pet boarding fee while I'm on official travel? | same cost; no underage drivers’ fee for drivers between the ages of 18-25; no minimum rental period; no cost for additional drivers; and full coverage insurance… | 0.1216 |
