"""Source of truth for the evaluation sets. Run this to regenerate
questions_test.jsonl and questions_dev.jsonl.

Written against the TTS Handbook "Travel & Leave" section BEFORE looking at
any model output.

Fields
  id, question
  answerable      False for questions the documents deliberately can't answer
  category        single_fact | reasoning | multi_hop | confusable |
                  unanswerable_tempting | unanswerable_out_of_scope
  stakes          high | medium | low  - cost of a wrong answer to an employee
  reference       the correct answer in plain words (for humans / LLM judge)
  key_facts       list of groups; the answer is "correct" only if it contains
                  at least one phrase from EVERY group (case-insensitive)
  evidence        exact substrings of the source text that support the answer;
                  retrieval is a "hit" when ALL of them appear in the top-k
  source          doc the evidence comes from
  trap            (unanswerable) what a model might wrongly say
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

TEST = [
    # ---------------------------------------------------------- leave -----
    dict(id="t01", question="How much annual leave can I carry over into the next leave year?",
         answerable=True, category="single_fact", stakes="high",
         reference="A maximum of 240 hours of annual leave.",
         key_facts=[["240"]],
         evidence=["carry over a maximum of 240 annual leave hours"], source="leave.md"),
    dict(id="t02", question="I've been a federal employee for 5 years. How much annual leave do I earn each pay period?",
         answerable=True, category="reasoning", stakes="medium",
         reference="6 hours per pay period (the 3-15 years bracket).",
         key_facts=[["6 hours", "six hours"]],
         evidence=["Between 3 and 15 years: 6 hours per pay period"], source="leave.md"),
    dict(id="t03", question="What is the most award leave I can receive in one fiscal year?",
         answerable=True, category="single_fact", stakes="low",
         reference="80 hours.",
         key_facts=[["80"]],
         evidence=["maximum of 80 hours of award leave in a single fiscal year"], source="leave.md"),
    dict(id="t04", question="If I leave TTS, will my unused award leave be paid out?",
         answerable=True, category="single_fact", stakes="medium",
         reference="No. Award leave is not paid out if you leave TTS.",
         key_facts=[["not paid out", "not be paid", "won't be paid", "will not be paid", "isn't paid", "is not paid", "no,"]],
         evidence=["If you leave TTS, award leave is not paid out"], source="leave.md"),
    dict(id="t05", question="How much sick leave can I use per year for a family member's funeral?",
         answerable=True, category="single_fact", stakes="medium",
         reference="Up to 13 days (104 hours) per year.",
         key_facts=[["13 days", "104 hours", "thirteen days"]],
         evidence=["13 days (104 hours) per year of your sick leave for bereavement"], source="leave.md"),
    dict(id="t06", question="How much parental bereavement leave do I get if my child passes away?",
         answerable=True, category="single_fact", stakes="high",
         reference="Two weeks (80 hours).",
         key_facts=[["two weeks", "2 weeks", "80 hours"]],
         evidence=["two weeks (80 hours) of parental bereavement leave"], source="leave.md"),
    dict(id="t07", question="How many days of military leave do I get each year as a National Guard member or Reservist?",
         answerable=True, category="single_fact", stakes="medium",
         reference="15 days per year; up to 15 days can carry over into the next fiscal year.",
         key_facts=[["15"]],
         evidence=["15 Military Leave days each year"], source="leave.md"),
    dict(id="t08", question="What disability rating do I need for disabled veteran leave, and how many hours can I get?",
         answerable=True, category="multi_hop", stakes="high",
         reference="A service-connected disability rating of 30% or more; up to 104 hours.",
         key_facts=[["30"], ["104"]],
         evidence=["30 percent or more", "up to 104 hours of disabled veteran leave"], source="leave.md"),
    dict(id="t09", question="If I'm approved for administrative leave to vote on election day, what is the maximum amount of time I can take?",
         answerable=True, category="single_fact", stakes="low",
         reference="Only the time necessary to vote, not to exceed 3 work hours.",
         key_facts=[["3 work hours", "3 hours", "three hours", "three work hours"]],
         evidence=["not to exceed 3 work hours"], source="leave.md"),
    dict(id="t10", question="What is the maximum number of credit hours I can have accrued at one time?",
         answerable=True, category="single_fact", stakes="low",
         reference="24 credit hours.",
         key_facts=[["24"]],
         evidence=["up to 24 credit hours accrued at any given time"], source="overtime.md"),
    dict(id="t11", question="When does regular comp time expire, and what happens to it if I leave GSA?",
         answerable=True, category="multi_hop", stakes="medium",
         reference="One year (26 pay periods) after it's earned; it is paid out if you leave GSA.",
         key_facts=[["one year", "1 year", "26 pay periods"], ["paid out"]],
         evidence=["Comp time expires one year (26 pay periods) after it's earned",
                   "If you leave GSA, your comp time will be paid out"], source="overtime.md"),
    dict(id="t12", question="What happens to my unused travel comp time if I leave GSA?",
         answerable=True, category="confusable", stakes="medium",
         reference="It is forfeited (unlike regular comp time, which is paid out).",
         key_facts=[["forfeit"]],
         evidence=["If you leave GSA, any unused travel comp time will be forfeited"], source="overtime.md"),
    dict(id="t13", question="What is the most advanced sick leave I can request?",
         answerable=True, category="confusable", stakes="high",
         reference="Up to 240 hours (term employees: up to what they'll earn for the rest of their term).",
         key_facts=[["240"]],
         evidence=["You can request up to 240 hours of advanced sick leave"], source="advanced-sick-leave.md"),
    dict(id="t14", question="How long do I need to have worked for the federal government to be eligible for FMLA?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least one year of federal service (most veterans are immediately eligible).",
         key_facts=[["one year", "1 year", "12 months"]],
         evidence=["you must have at least one year of federal service"], source="fmla.md"),
    dict(id="t15", question="If I take all 12 weeks of paid parental leave, how much FMLA do I have left for the next 52 weeks?",
         answerable=True, category="reasoning", stakes="high",
         reference="None. Maxing out 12 weeks of PPL uses up all FMLA for the next 52 weeks.",
         key_facts=[["uses up all", "use up all", "none", "no fmla", "0 weeks", "zero", "no remaining", "used up"]],
         evidence=["If you max out 12 weeks of PPL, that uses up all the FMLA you can use for the next 52 weeks"], source="fmla.md"),
    dict(id="t16", question="After paid parental leave, how long do I have to agree to return to work?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least 12 weeks after your final day of PPL.",
         key_facts=[["12 weeks", "twelve weeks"]],
         evidence=["return to work for at least 12 weeks after your final day on PPL"], source="paid-parental-leave.md"),
    dict(id="t17", question="How much bereavement leave can I use per year for a civilian family member, and how much funeral leave do I get for an immediate relative's combat-related death?",
         answerable=True, category="multi_hop", stakes="medium",
         reference="Up to 13 days (104 hours) of sick leave per year; 3 days (24 hours) of funeral leave for a combat-related death.",
         key_facts=[["13 days", "104 hours"], ["3 days", "24 hours", "three days"]],
         evidence=["13 days (104 hours) per year of your sick leave for bereavement",
                   "3 days (24 hours) of funeral leave"], source="leave.md"),
    # --------------------------------------------------------- travel -----
    dict(id="t18", question="Above what dollar amount do I need a receipt for a travel expense?",
         answerable=True, category="single_fact", stakes="medium",
         reference="Any expense over $75 (plus lodging, rental cars and rental-car gas regardless of amount).",
         key_facts=[["75"]],
         evidence=["any other expenses over $75"], source="travel-guide-3-travel.md"),
    dict(id="t19", question="What percentage of the normal M&IE rate do I get on travel days?",
         answerable=True, category="single_fact", stakes="low",
         reference="75% of normal M&IE.",
         key_facts=[["75"]],
         evidence=["always 75% of normal M&IE"], source="travel-guide-3-travel.md"),
    dict(id="t20", question="How soon after returning from a trip should I submit my travel voucher?",
         answerable=True, category="single_fact", stakes="medium",
         reference="Within 5 business days of getting back.",
         key_facts=[["5 business days", "five business days"]],
         evidence=["within 5 business days of getting back"], source="travel-guide-3-travel.md"),
    dict(id="t21", question="Can supervisors get reimbursed for professional liability insurance, and how much?",
         answerable=True, category="single_fact", stakes="low",
         reference="Yes: up to half the cost, up to $150.",
         key_facts=[["150"], ["half"]],
         evidence=["up to half the cost of professional liability insurance, up to $150"], source="travel-guide-faq.md"),
    dict(id="t22", question="How far in advance do I need to contact the Travel and Events team for official international travel?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least 7 weeks in advance (international travel needs Salesforce event approval).",
         key_facts=[["7 weeks", "seven weeks"]],
         evidence=["at least 7 weeks in advance in order to coordinate this"], source="travel-guide-faq.md"),
    dict(id="t23", question="What counts as a 'large team gathering' that needs the event request process instead of an approval email?",
         answerable=True, category="single_fact", stakes="medium",
         reference="Over 6 employees traveling for an internal management meeting and/or more than $10,000 in estimated group travel expenses.",
         key_facts=[["6"], ["10,000", "10000", "$10k"]],
         evidence=["over 6 employees traveling for an internal management meeting"], source="travel-guide-faq.md"),

    # ---------------------------------------------------- unanswerable ----
    dict(id="u01", question="What is the maximum per diem lodging rate for a hotel in Chicago?",
         answerable=False, category="unanswerable_tempting", stakes="medium",
         reference="Not in the documents. They say to stay under the per diem rate and link to a lookup, but list no Chicago rate.",
         trap="A specific dollar figure from training data.", source=None),
    dict(id="u02", question="What is the daily meals and incidental expenses rate in Boston?",
         answerable=False, category="unanswerable_tempting", stakes="medium",
         reference="Not in the documents. The only M&IE figure is an example for DC ($69); 'Boston' appears only in a Slack-status example.",
         trap="Reusing the DC $69 example, or a Boston figure from memory.", source=None),
    dict(id="u03", question="What is the reimbursement rate per mile for using my personal car on official travel?",
         answerable=False, category="unanswerable_tempting", stakes="medium",
         reference="Not in the documents. They mention 'POV mileage rates' but never state a rate.",
         trap="The IRS/GSA mileage rate from memory (e.g. ~70 cents/mile).", source=None),
    dict(id="u04", question="Can I fly business class on an international flight longer than 14 hours?",
         answerable=False, category="unanswerable_tempting", stakes="high",
         reference="Not in the documents. The handbook doesn't cover premium-class travel.",
         trap="'Yes, the FTR allows premium class for flights over 14 hours' (outside knowledge).", source=None),
    dict(id="u05", question="How many paid federal holidays do GSA employees get each year?",
         answerable=False, category="unanswerable_out_of_scope", stakes="low",
         reference="Not in the documents.",
         trap="'11 federal holidays' from general knowledge.", source=None),
    dict(id="u06", question="How much does GSA match on my TSP retirement contributions?",
         answerable=False, category="unanswerable_out_of_scope", stakes="high",
         reference="Not in the documents. TSP is only mentioned in the context of updating your address after a move.",
         trap="'Up to 5%' from general knowledge.", source=None),
    dict(id="u07", question="Can I expense a pet boarding fee while I'm on official travel?",
         answerable=False, category="unanswerable_tempting", stakes="low",
         reference="Not in the documents.",
         trap="A confident yes/no extrapolated from 'other necessary expenses'.", source=None),
]

# Dev set: used ONLY to calibrate the refusal threshold, never to report results.
DEV = [
    dict(id="d01", question="How much administrative leave can I take to be a bone marrow donor?",
         answerable=True, category="single_fact", stakes="medium", reference="Up to 7 days per calendar year.",
         key_facts=[["7 days", "seven days"]], evidence=["up to 7 days of Administrative Leave"], source="leave.md"),
    dict(id="d02", question="How many nights do I need to be traveling before laundry is reimbursable?",
         answerable=True, category="single_fact", stakes="low", reference="At least 4 consecutive nights.",
         key_facts=[["4", "four"]], evidence=["at least 4 consecutive nights"], source="travel-guide-3-travel.md"),
    dict(id="d03", question="If I relocate temporarily for a couple of months, do I need to update my address in GSA systems?",
         answerable=True, category="reasoning", stakes="low", reference="No, not if it's 120 days or less.",
         key_facts=[["120"]], evidence=["120 days or less, you do not need to update GSA systems"], source="moving.md"),
    dict(id="d04", question="What size of rental car am I required to book?",
         answerable=True, category="single_fact", stakes="low", reference="The lowest cost compact car unless approved for larger.",
         key_facts=[["compact"]], evidence=["lowest cost compact car"], source="travel-guide-faq.md"),
    dict(id="d05", question="Do I need to have used up my annual leave to receive donated leave through VLTP?",
         answerable=True, category="single_fact", stakes="medium", reference="Yes, all leave balances must be exhausted.",
         key_facts=[["exhaust"]], evidence=["Exhausted all leave balances"], source="voluntary-leave-transfer-program.md"),
    dict(id="d06", question="How soon after returning from personal foreign travel do I need to complete the debriefing form?",
         answerable=True, category="single_fact", stakes="medium", reference="Within five days.",
         key_facts=[["five days", "5 days"]], evidence=["Within five days of your return from foreign travel"], source="unofficial-foreign-travel.md"),
    dict(id="d07", question="What is the maximum hotel rate in San Francisco?",
         answerable=False, category="unanswerable_tempting", stakes="medium", reference="Not in the documents.", source=None),
    dict(id="d08", question="Does GSA reimburse gym fees while I'm traveling?",
         answerable=False, category="unanswerable_tempting", stakes="low", reference="Not in the documents.", source=None),
    dict(id="d09", question="How much life insurance coverage does FEGLI provide?",
         answerable=False, category="unanswerable_out_of_scope", stakes="medium", reference="Not in the documents.", source=None),
]


def write(rows, name):
    with open(HERE / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {name} "
          f"({sum(not r['answerable'] for r in rows)} unanswerable)")


if __name__ == "__main__":
    write(TEST, "questions_test.jsonl")
    write(DEV, "questions_dev.jsonl")
