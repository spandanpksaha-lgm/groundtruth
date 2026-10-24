"""Evaluation sets for the UC Student Handbook 2026-2027 corpus.
Run this to regenerate questions_uc_test.jsonl and questions_uc_dev.jsonl.

Written by Claude from a full read of the handbook, before any model output
was seen. Same fields as eval/make_questions.py, plus one extra category:

  conflicting  the handbook itself gives two different answers in two places.
               A good answer surfaces both; a confident single answer is
               graded wrong. These probe a real risk of RAG over policy
               documents: the source can be inconsistent.

Evidence strings are exact text from the handbook (checked by
eval/verify_questions.py). Page numbers refer to the PDF.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

NOT_ALLOWED = ["not permitted", "not allowed", "prohibited", "may not", "cannot", "can't", "no,", "isn't allowed",
               "are not", "is not"]

TEST = [
    # -------------------------------------------------------- academics ----
    dict(id="uc_t01", question="How long after the last day of class do I have to file a formal written appeal of a grade?",
         answerable=True, category="single_fact", stakes="high",
         reference="Four weeks after the last day of class for that semester (after first talking to the instructor and department chair). p.13",
         key_facts=[["four weeks", "4 weeks"]],
         evidence=["must be filed four weeks after the last day of class"], source="p.13 Academic Appeals"),
    dict(id="uc_t02", question="I got seriously ill and can't attend classes. How soon do I have to submit a medical withdrawal request?",
         answerable=True, category="single_fact", stakes="high",
         reference="Within three weeks of becoming unable to attend, with supporting documentation from a medical professional. p.13",
         key_facts=[["three weeks", "3 weeks"]],
         evidence=["within three weeks of becoming unable to attend classes"], source="p.13 Medical / Emergency Withdrawal"),
    dict(id="uc_t03", question="How many hours of community service do main campus undergraduates need to graduate?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least 40 hours. p.12",
         key_facts=[["40", "forty"]],
         evidence=["minimum of 40 hours of community service to graduate"], source="p.12 Leadership & Community Service Program"),
    dict(id="uc_t04", question="How many community service hours do I need to be recognized as a Hutton Scholar?",
         answerable=True, category="single_fact", stakes="low",
         reference="200 or more hours (veterans and active service members receive 200 automatically). p.13",
         key_facts=[["200"]],
         evidence=["complete and report 200+ community service hours"], source="p.13 Hutton Scholar Program"),
    dict(id="uc_t05", question="I'm an on-campus student. During the fall semester, how many in-seat hours must I take before I can register for 6 hours of online classes?",
         answerable=True, category="reasoning", stakes="medium",
         reference="At least 9 semester hours of in-seat courses. p.12",
         key_facts=[["9", "nine"]],
         evidence=["enroll in at least 9 semester hours of in-seat courses"], source="p.12 On-Campus Student Eligibility for Online Courses"),
    # ---------------------------------------------------- financial aid ----
    dict(id="uc_t06", question="What is the pay rate for most Federal Work-Study jobs on campus?",
         answerable=True, category="single_fact", stakes="low",
         reference="$8.50/hr for most jobs; food service and plant service jobs pay $9.50/hr. p.15",
         key_facts=[["8.50"]],
         evidence=["The pay rate is $8.50/hr for most jobs"], source="p.15 Federal Work-study"),
    dict(id="uc_t07", question="How many credit hours do I need each semester to count as a full-time undergraduate for financial aid?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least 12 credit hours. p.18",
         key_facts=[["12", "twelve"]],
         evidence=["at least 12 credit hours each semester to be considered full-time"], source="p.18 Satisfactory Academic Progress"),
    dict(id="uc_t08", question="What cumulative GPA must an undergraduate keep to maintain Satisfactory Academic Progress for financial aid?",
         answerable=True, category="single_fact", stakes="high",
         reference="At least 2.0 on a 4.0 scale. p.19",
         key_facts=[["2.0"]],
         evidence=["All undergraduate students must maintain a cumulative GPA of at least a 2.0"], source="p.19 Satisfactory Academic Progress"),
    dict(id="uc_t09", question="My undergraduate degree requires 120 credit hours. What's the maximum number of hours I can attempt and still receive financial aid?",
         answerable=True, category="reasoning", stakes="high",
         reference="180 hours (150% of 120). p.19",
         key_facts=[["180"]],
         evidence=["attempt a maximum of 180"], source="p.19 Satisfactory Academic Progress"),
    dict(id="uc_t10", question="What share of the credit hours I attempt do I have to actually earn to keep my financial aid?",
         answerable=True, category="single_fact", stakes="medium",
         reference="67% of all hours attempted; W, I and F grades count as attempted but not earned. p.19",
         key_facts=[["67"]],
         evidence=["earn credits in 67% of all hours they attempt"], source="p.19 Satisfactory Academic Progress"),
    dict(id="uc_t11", question="If I fail to meet Satisfactory Academic Progress after one semester, do I lose my financial aid right away?",
         answerable=True, category="single_fact", stakes="high",
         reference="No. You are placed on Financial Aid Warning and stay eligible for one semester; failing again leads to SAP Suspension. p.19",
         key_facts=[["warning"]],
         evidence=["Financial Aid Warning"], source="p.19 Satisfactory Academic Progress"),
    # -------------------------------------------------------- parking -----
    dict(id="uc_t12", question="How long do I have to dispute a campus parking ticket?",
         answerable=True, category="single_fact", stakes="medium",
         reference="Within 7 days of the issue date, through the parking portal. p.61",
         key_facts=[["7 days", "seven days", "seven (7) days"]],
         evidence=["within 7 days of the issue date"], source="p.61 Vehicle Fines & Disputes"),
    dict(id="uc_t13", question="This is my third parking ticket this academic year. How much will the fine be?",
         answerable=True, category="reasoning", stakes="low",
         reference="$30: fines start at $20 and rise $5 per additional violation in the same academic year. p.61",
         key_facts=[["30"]],
         evidence=["third = $30.00"], source="p.61 Escalating Fines"),
    dict(id="uc_t14", question="How many parking tickets in one semester get my car towed, and how much does it cost to get it back?",
         answerable=True, category="multi_hop", stakes="medium",
         reference="4 or more citations in one semester; retrieval costs $65, paid to the towing company. p.61",
         key_facts=[["4", "four"], ["65"]],
         evidence=["(4) or more citations within (1) semester", "$65.00"], source="p.61 Parking Violations"),
    dict(id="uc_t15", question="What's the speed limit for electric scooters on the campus shared-use paths?",
         answerable=True, category="single_fact", stakes="low",
         reference="10 mph on designated shared-use paths (15 mph on campus roads). p.58",
         key_facts=[["10"]],
         evidence=["10 mph on designated shared-use paths"], source="p.58 Personal Transportation Device Policy"),
    # --------------------------------------------------- residence life ----
    dict(id="uc_t16", question="If I'm approved to stay in my residence hall over a break, how much will I be charged?",
         answerable=True, category="single_fact", stakes="low",
         reference="$20 per day (may be waived for athletics, campus work, or at the University's discretion). p.67",
         key_facts=[["20"]],
         evidence=["charged a fee at the rate of $20 per day"], source="p.67 Break Period Checkout"),
    dict(id="uc_t17", question="What's the most powerful microwave I'm allowed to keep in my residence hall room?",
         answerable=True, category="single_fact", stakes="low",
         reference="700 watts or less (refrigerators 3.3 cubic feet or less). p.70",
         key_facts=[["700"]],
         evidence=["microwave ovens (700 watts or less)"], source="p.70 Refrigerators/Microwaves"),
    dict(id="uc_t18", question="How many nights per semester can an overnight guest stay with me in the residence hall?",
         answerable=True, category="single_fact", stakes="low",
         reference="A maximum of two nights per semester, unless the Director of Residence Life approves more. p.72",
         key_facts=[["two", "2"]],
         evidence=["maximum of two nights per semester"], source="p.72 Overnight Guests"),
    dict(id="uc_t19", question="When do quiet hours start and end on a Friday night in the residence halls?",
         answerable=True, category="single_fact", stakes="low",
         reference="11:00 pm until 9:00 am. p.75",
         key_facts=[["11"], ["9"]],
         evidence=["Friday and Saturday from 11:00pm until 9:00am"], source="p.75 Quiet Hours"),
    dict(id="uc_t20", question="Can I use an air fryer in my dorm room? What about a coffee maker?",
         answerable=True, category="confusable", stakes="medium",
         reference="Air fryers are not permitted (heating element). Coffee makers are permitted but must be plugged directly into a wall outlet, never a power strip. p.68",
         key_facts=[NOT_ALLOWED, ["coffee maker"]],
         evidence=["crock pots, air fryers, and rice", "Coffee makers, blow dryers, hair curlers"], source="p.68 Prohibited Items"),
    # -------------------------------------------------------- services ----
    dict(id="uc_t21", question="How long does the university hold lost-and-found items before giving them away?",
         answerable=True, category="single_fact", stakes="low",
         reference="30 days, then items go to a charitable organization. p.66",
         key_facts=[["30", "thirty"]],
         evidence=["not claimed within a period of thirty (30) days"], source="p.66 Lost and Found"),
    dict(id="uc_t22", question="How long will my university email account stay active after I graduate?",
         answerable=True, category="single_fact", stakes="medium",
         reference="About 18 months (up to three years for F-1 students on OPT). p.22",
         key_facts=[["18"]],
         evidence=["approximately 18 months following graduation"], source="p.22 University Account"),
    dict(id="uc_t23", question="What time does the library close on Fridays?",
         answerable=True, category="single_fact", stakes="low",
         reference="11:00 pm (Friday hours 8:00 am-11:00 pm). Campus Facilities Schedule",
         key_facts=[["11"]],
         evidence=["Friday 8:00 am-11:00 pm"], source="Campus Facilities Schedule"),
    # ----------------------------------------- the handbook contradicts itself
    dict(id="uc_t24", question="Am I allowed to ride my skateboard on campus sidewalks?",
         answerable=True, category="conflicting", stakes="medium",
         reference="The handbook conflicts. The Personal Transportation Device Policy (p.57-58) says riding is NOT permitted on sidewalks; the older Skateboard/Hoverboard Policy (p.66) says skateboards are to be used on sidewalks only. A good answer reports both.",
         key_facts=[NOT_ALLOWED, ["sidewalks only", "only on sidewalks", "on sidewalks only", "conflict", "contradict",
                                  "inconsisten", "another section", "other section", "also says", "however"]],
         evidence=["On sidewalks or in pedestrian-only areas", "are to be used on sidewalks only"],
         source="p.58 Personal Transportation Device Policy vs p.66 Skateboard/Hoverboard Policy"),
    dict(id="uc_t25", question="Which online alcohol-education course do first-year students have to complete?",
         answerable=True, category="conflicting", stakes="low",
         reference="The handbook names two: 3rd Millennium Classrooms' Alcohol Wise (p.27) and AlcoholEdu for College through Vector Solutions (p.48), both as part of OREN 100. A good answer reports both.",
         key_facts=[["alcohol wise", "alcoholwise"], ["alcoholedu"]],
         evidence=["Alcohol Wise is a comprehensive", "utilizes AlcoholEdu for College"],
         source="p.27 Alcohol and Drug Education vs p.48 Prevention and Education"),

    # --------------------------------------------------- unanswerable ----
    dict(id="uc_u01", question="How much is undergraduate tuition per credit hour?",
         answerable=False, category="unanswerable_tempting", stakes="high",
         reference="Not in the handbook. The only tuition figure ($4,938) is part of a worked financial-aid example, not a price list.",
         trap="Quoting the $4,938 from the Return of Title IV example as the tuition rate.", source=None),
    dict(id="uc_u02", question="What GPA do I need to graduate cum laude?",
         answerable=False, category="unanswerable_out_of_scope", stakes="medium",
         reference="Not in the handbook (academic honors are in the University Catalog).",
         trap="A GPA such as 3.5 from general knowledge.", source=None),
    dict(id="uc_u03", question="How many absences am I allowed in an in-seat class before I fail?",
         answerable=False, category="unanswerable_tempting", stakes="high",
         reference="Not stated. The handbook says participation is monitored in the first two weeks and that penalties are in each course syllabus; the one numeric rule is for synchronous online sessions.",
         trap="Applying the online 'more than one weekly synchronous session' rule, or a number from memory.", source=None),
    dict(id="uc_u04", question="How much does a single room in the residence halls cost per semester?",
         answerable=False, category="unanswerable_out_of_scope", stakes="medium",
         reference="Not in the handbook.", trap="A dollar figure from general knowledge.", source=None),
    dict(id="uc_u05", question="How much is the late fee if I miss my tuition payment due date?",
         answerable=False, category="unanswerable_tempting", stakes="medium",
         reference="Not stated. The handbook says the account 'is subject to a late fee' but gives no amount.",
         trap="Inventing an amount, or quoting the $20 parking fine.", source=None),
    dict(id="uc_u06", question="How many meal swipes per week come with the resident meal plan?",
         answerable=False, category="unanswerable_tempting", stakes="low",
         reference="Not stated. The handbook says resident students participate in the meal plan and lists dining hall hours, but not swipes per week.",
         trap="A number such as 14 or 19 from general knowledge.", source=None),
    dict(id="uc_u07", question="How much does it cost to replace a lost student ID card?",
         answerable=False, category="unanswerable_tempting", stakes="low",
         reference="Not stated. The handbook says a replacement must be purchased from the Office of Student Affairs but gives no price.",
         trap="Quoting the $50 key-replacement fee.", source=None),
]

# Dev set: ONLY for calibrating the refusal threshold, never for reporting.
DEV = [
    dict(id="uc_d01", question="What's the fine for skipping a mandatory residence hall meeting?",
         answerable=True, category="single_fact", stakes="low", reference="$50 per meeting. p.71",
         key_facts=[["50"]], evidence=["will result in a $50.00 fine for each meeting"], source="p.71"),
    dict(id="uc_d02", question="How much do off-campus visitors aged 13 and over pay to use the Aquatic Center?",
         answerable=True, category="single_fact", stakes="low", reference="$4.00 per visit. p.64",
         key_facts=[["4.00", "$4"]], evidence=["13 and over pay $4.00 per visit"], source="p.64"),
    dict(id="uc_d03", question="What is the highest deductible a student-athlete's own insurance can have to waive the university accident plan?",
         answerable=True, category="single_fact", stakes="medium", reference="$10,000 per year. p.66",
         key_facts=[["10,000"]], evidence=["it must not exceed $10,000 per year"], source="p.66"),
    dict(id="uc_d04", question="How many credit hours must I have completed to live in the undergraduate apartments?",
         answerable=True, category="single_fact", stakes="medium", reference="At least 30. p.69",
         key_facts=[["30"]], evidence=["minimum of 30 credit hours completed"], source="p.69"),
    dict(id="uc_d05", question="How long do I have to appeal a decision made after an individual threat assessment?",
         answerable=True, category="single_fact", stakes="high", reference="Ten calendar days, in writing to the VP for Student Affairs. p.47",
         key_facts=[["10", "ten"]], evidence=["within ten (10) calendar days after the decision"], source="p.47"),
    dict(id="uc_d06", question="How far in advance do I need to request the free airport shuttle?",
         answerable=True, category="single_fact", stakes="low", reference="At least 14 days. p.76",
         key_facts=[["14"]], evidence=["at least 14 days in advance"], source="p.76"),
    dict(id="uc_d07", question="What is the Wi-Fi password for the residence halls?",
         answerable=False, category="unanswerable_tempting", stakes="low", reference="Not in the handbook.", source=None),
    dict(id="uc_d08", question="What is the dress code for the graduation ceremony?",
         answerable=False, category="unanswerable_out_of_scope", stakes="low", reference="Not in the handbook.", source=None),
    dict(id="uc_d09", question="How many students live on campus?",
         answerable=False, category="unanswerable_out_of_scope", stakes="low", reference="Not in the handbook.", source=None),
]


def write(rows, name):
    with open(HERE / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {name} ({sum(not r['answerable'] for r in rows)} unanswerable)")


if __name__ == "__main__":
    write(TEST, "questions_uc_test.jsonl")
    write(DEV, "questions_uc_dev.jsonl")
