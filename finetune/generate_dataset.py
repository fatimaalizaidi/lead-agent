"""Synthetic training data for the priority classifier (seeded, reproducible, no API calls).

Each lead is built from templates with random slots (company, seats, budget, timeline, feature,
competitor, greeting, sign-off) plus light noise (lowercase, missing punctuation, typos), and
labelled with the same rubric as ``main.SYSTEM_PROMPT``:

- **High**: clear buying intent (budget, timeline, seats, urgency).
- **Medium**: interested but exploratory.
- **Low**: vague, spam, or poor fit.

The held-out test set (``finetune/data/test.jsonl``) is hand-written and never generated here,
so test accuracy measures whether the student generalises beyond these templates. To label with
the production LLM instead of the rubric (true teacher → student distillation), run
``python -m finetune.label_with_teacher`` on the output.

Usage:  python -m finetune.generate_dataset            # writes train.jsonl + val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LABELS = ("High", "Medium", "Low")

COMPANIES = [
    "Acme", "Northwind", "Globex", "Initech", "Umbrella Health", "Stark Logistics", "Wayne Retail",
    "Hooli", "Pied Piper", "Vandelay Imports", "Soylent Foods", "Cyberdyne", "Tyrell Labs", "Oscorp",
    "Wonka Industries", "Gringotts Finance", "Monarch Realty", "Blue Sun", "Massive Dynamic", "Aperture",
    "Lumon", "Dunder Mifflin", "Bluth Homes", "Prestige Worldwide", "Sterling Cooper", "Kingsford Legal",
    "Harbor Dental", "Peak Fitness", "Orbit Travel", "Crescent Bank", "Falcon Freight", "Maple Schools",
]
ROLES = [
    "Head of Sales", "VP Revenue", "RevOps lead", "Sales Director", "CTO", "founder", "COO",
    "marketing manager", "procurement lead", "Head of Growth", "CEO", "sales ops manager",
]
FEATURES = [
    "lead scoring", "the n8n integration", "Slack alerts", "the CRM sync", "SSO", "the API",
    "auto-drafted replies", "priority routing", "the analytics dashboard", "multi-language support",
    "data retention controls", "the Salesforce connector", "HubSpot sync", "webhooks",
]
COMPETITORS = ["HubSpot", "Salesforce Einstein", "Clearbit", "Apollo", "Outreach", "Drift", "a spreadsheet"]
TIMELINES = [
    "this week", "by Friday", "before the end of the month", "by the 1st", "within two weeks",
    "before our Q1 kickoff", "by next Monday", "ASAP", "before our contract with our current vendor ends",
    "this quarter", "in the next 10 days",
]
PLANS = ["Starter", "Growth", "Enterprise", "the annual plan", "Growth annual"]
GREETINGS = ["", "Hi,", "Hello,", "Hi team,", "Hey there,", "Good morning,", "Dear sales team,", "Hi folks,"]
SIGNOFFS = ["", "Thanks", "Thanks!", "Best,", "Regards", "Cheers", "Thank you."]

HIGH = [
    "We have budget approved for {seats} seats and want to start {timeline}. Can you send a contract?",
    "I'm the {role} at {company}. We need {plan} for {seats} reps {timeline}. What's the next step to sign?",
    "Ready to buy. {seats} seats on {plan}, budget is ${budget}k for the year. Who do I talk to about invoicing?",
    "Our current tool is being shut down and we need a replacement {timeline}. {seats} users, budget already signed off.",
    "Please send a quote for {seats} seats on {plan}. Procurement wants to issue the PO {timeline}.",
    "We'd like to move forward with {plan}. Can we get a call {timeline} to finalise pricing for {seats} seats?",
    "{company} is switching from {competitor}. We have ${budget}k allocated and want to go live {timeline}.",
    "Urgent: our team of {seats} needs {feature} live {timeline}. Can you onboard us that fast? Budget is not an issue.",
    "We tested the trial and the team loves it. Let's upgrade to {plan} for {seats} seats, starting {timeline}.",
    "Can you do {seats} seats at an annual commitment? We'd sign {timeline} if the price works for a ${budget}k budget.",
    "Looking to purchase {plan} for {company}. {seats} seats, need SSO, decision is made, just need the paperwork.",
    "Our board approved ${budget}k for lead tooling. We want {plan} for {seats} seats and can start {timeline}.",
    "I need a signed order form {timeline}. {seats} seats on {plan}, billed annually. Who can send it?",
    "We're expanding from 5 to {seats} seats {timeline}. Please upgrade our account and send the invoice.",
    "Can someone call me today? We want to roll out to {seats} sales reps {timeline} and need {feature} configured.",
    "Final step for us is security review. If you can send the SOC 2 report we'll sign for {seats} seats {timeline}.",
    "Please book a demo for our {seats}-person team {timeline}; we plan to buy right after if {feature} works.",
    "We're replacing {competitor} {timeline}. {seats} seats, ${budget}k budget, need a quote today.",
    "Our renewal with {competitor} ends {timeline}. We want to switch {seats} seats to {plan}. How fast can we start?",
    "As {role} I can sign off up to ${budget}k. Send me pricing for {seats} seats and we'll purchase {timeline}.",
]
MEDIUM = [
    "How does your {feature} compare to {competitor}? We're looking at options for next year.",
    "Just exploring tools for our sales team. What does {plan} include?",
    "Could you share more information about {feature}? We might be interested later.",
    "We're evaluating a few vendors. Do you have a case study for companies like {company}?",
    "What integrations do you support? Thinking about {feature} for our team at some point.",
    "Is there a free trial? I'd like to play around with it before talking to my manager.",
    "I'm researching lead tools for {company}. How is pricing calculated?",
    "Do you offer {feature}? Not urgent, just building a shortlist.",
    "We might need something like this in a few months. Can you send a brochure?",
    "Curious how your AI drafts replies. Does it work in languages other than English?",
    "Our {role} asked me to look into lead scoring tools. What makes yours different from {competitor}?",
    "Do you have a webinar or a recorded demo I could watch?",
    "Thinking about switching from {competitor} eventually. What's the migration process like?",
    "We're a team of {seats} and growing. Would {plan} make sense for us down the line?",
    "No budget yet, but I'd love to understand how {feature} works.",
    "Can you explain your refund policy? We want to know before we consider a trial.",
    "What are your support hours? Just gathering info for an internal comparison.",
    "Does {plan} include {feature}, or is that Enterprise only? Comparing plans for now.",
    "Interested in learning more. Is there documentation for {feature}?",
    "We're early in our search. What's your typical onboarding time for a company like {company}?",
]
LOW = [
    "hi",
    "hello?",
    "test",
    "asdfgh",
    "We offer cheap SEO backlinks and guest posts. Reply for our price list!",
    "Congratulations, you've won a gift card! Click here to claim.",
    "I'm a student writing a thesis on AI. Can you answer a few questions?",
    "Please unsubscribe me from your emails.",
    "Are you hiring? I'd like to apply for a sales job.",
    "We build mobile apps for startups at low rates. Interested in a partnership?",
    "Can you help me fix my printer?",
    "I'm looking for a free tool for my personal blog, not for a business.",
    "Do you sell office furniture?",
    "Buy followers for Instagram, 10k for $5, DM us.",
    "My cousin told me about you. What do you do exactly?",
    "Is this a crypto investment platform?",
    "Hey, just saying hi. Nice website.",
    "I want to sell you my email list of 2 million contacts.",
    "Need a recipe for chocolate cake.",
    "Your website is slow. We can redesign it for $299.",
    "Wrong number, sorry.",
    "I'm a one-person hobby project and don't need sales tools, just curious.",
    "Can I get a job reference letter?",
    "Win big at our online casino, sign up bonus 200%!",
    "?",
    "Looking for a plumber in Austin.",
    "We are an offshore outsourcing agency. Can we schedule a call to pitch our services?",
    "Please stop calling me.",
    "lol",
    "Do you have any internships for the summer?",
    # Slot-filled Low leads, so company names, seat counts and features aren't High/Medium shortcuts.
    "I'm from {company}'s marketing agency. We can get you {seats} new clients a month, guaranteed!",
    "Hi, {role} at {company} here. Want to buy our list of {seats}k verified leads?",
    "We resell {competitor} licences at 70% off. Interested in becoming a partner?",
    "Can you write my college essay about {feature}? Will pay $20.",
    "Not interested, please remove {company} from your mailing list.",
    "I run a {seats}-person bakery, do you deliver bread?",
    "Your {feature} page has a typo. Anyway, do you buy guest posts?",
    "We're {company}, a web design agency. Can we redesign your site {timeline}?",
]


def _noise(text: str, rng: random.Random) -> str:
    """Light, label-preserving noise so the student doesn't key on punctuation or casing."""
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    elif roll < 0.22:
        text = text.rstrip(".!?")
    if rng.random() < 0.08 and len(text) > 20:
        i = rng.randrange(1, len(text) - 1)
        text = text[:i] + text[i + 1 :]  # drop one character
    return re.sub(r"\s+", " ", text).strip()


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        seats=rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200]),
        budget=rng.choice([5, 10, 15, 20, 25, 40, 50, 80, 120]),
        timeline=rng.choice(TIMELINES),
        role=rng.choice(ROLES),
        company=rng.choice(COMPANIES),
        plan=rng.choice(PLANS),
        feature=rng.choice(FEATURES),
        competitor=rng.choice(COMPETITORS),
    )


def make_lead(label: str, rng: random.Random) -> str:
    template = rng.choice({"High": HIGH, "Medium": MEDIUM, "Low": LOW}[label])
    body = _fill(template, rng)
    # Spam and one-word messages don't get a greeting/sign-off wrapper.
    if label != "Low" or len(body) > 40:
        body = " ".join(p for p in (rng.choice(GREETINGS), body, rng.choice(SIGNOFFS)) if p)
    return _noise(body, rng)


def generate(n_per_label: int, seed: int, exclude: set[str]) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen = set(exclude)
    for label in LABELS:
        attempts = 0
        count = 0
        while count < n_per_label and attempts < n_per_label * 50:
            attempts += 1
            text = make_lead(label, rng)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label})
            count += 1
    rng.shuffle(rows)
    return rows


def read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-per-label", type=int, default=400)
    parser.add_argument("--val-per-label", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    test_texts = {r["text"].lower() for r in read_jsonl(DATA_DIR / "test.jsonl")}
    rows = generate(args.train_per_label + args.val_per_label, args.seed, exclude=test_texts)
    val: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    per_label_val = dict.fromkeys(LABELS, 0)
    for row in rows:
        if per_label_val[row["label"]] < args.val_per_label:
            per_label_val[row["label"]] += 1
            val.append(row)
        else:
            train.append(row)
    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    print(f"train={len(train)} val={len(val)} (test is hand-written: {len(test_texts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
