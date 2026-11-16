# Priority classifier: hand-written test set (84 leads)

| Model | Trainable params | accuracy | macro_f1 | high_recall | high_precision | low_recall | latency_ms_p50 | size_mb |
|---|---|---|---|---|---|---|---|---|
| lora (model) | 74,883 / 22,789,254 | 0.8571 | 0.8545 | 1.0 | 0.8333 | 0.931 | 2.46 | 23.0 |
| head-only (priority-head-only) | 1,155 / 22,714,371 | 0.7262 | 0.7301 | 0.72 | 0.8182 | 0.6207 | 2.44 | 23.0 |

## lora (model)

Confusion matrix (rows = true label, columns = predicted):

| | High | Medium | Low |
|---|---|---|---|
| **High** | 25 | 0 | 0 |
| **Medium** | 5 | 20 | 5 |
| **Low** | 0 | 2 | 27 |

Misclassified:

- `Medium` → `Low` (0.43): Hello! I run sales at a 40-person agency and we're shopping around for lead triage software. What does Growth cost?
- `Medium` → `Low` (0.62): Saw your post on LinkedIn. Looks interesting, could you tell me more about how it works?
- `Medium` → `Low` (0.53): Our marketing team is curious whether this could help with webinar leads. Any examples?
- `Medium` → `Low` (0.52): Do you support Arabic and Urdu messages? We might expand to new markets eventually.
- `Medium` → `High` (0.86): We're a 12-person startup. Maybe next quarter we'll need something like this. What's the cheapest plan?
- `Medium` → `Low` (0.50): Is there a nonprofit discount? We're considering options for 2027.
- `Medium` → `High` (0.88): How do refunds work if we decide it's not a fit after a month?
- `Medium` → `High` (0.40): Can your system handle 5,000 leads a month? Just sizing things up.
- `Medium` → `High` (0.96): How does billing work for seats added mid-month? Curious before we look deeper.
- `Medium` → `High` (0.88): Hi! Could someone explain what a 'seat' means in your pricing? Trying to estimate costs.
- `Low` → `Medium` (0.44): Testing the form, ignore this.
- `Low` → `Medium` (0.51): asdf asdf asdf

## head-only (priority-head-only)

Confusion matrix (rows = true label, columns = predicted):

| | High | Medium | Low |
|---|---|---|---|
| **High** | 18 | 6 | 1 |
| **Medium** | 3 | 25 | 2 |
| **Low** | 1 | 10 | 18 |

Misclassified:

- `High` → `Medium` (0.34): We're churning off our current vendor at month end. Need 60 licences, annual billing. Who handles contracts?
- `High` → `Medium` (0.39): I'd like to buy the Starter plan for my team of three right now. Where do I enter the card?
- `High` → `Medium` (0.38): Need to replace our lead router urgently, the old one broke our pipeline. 18 users. Can you start tomorrow?
- `High` → `Medium` (0.37): Our contract with HubSpot expires on the 31st. Moving 70 users over, budget approved. Need a migration plan.
- `High` → `Low` (0.36): I'm the VP Sales at a fintech, 90 reps. We've chosen you. What do you need from us to start this month?
- `High` → `Medium` (0.44): We want the annual Enterprise contract. Our security team finished the review. Who sends the MSA?
- `High` → `Medium` (0.38): Decision made, we're buying. Just need the Enterprise pricing sheet for 80 users by tomorrow.
- `Medium` → `High` (0.38): How long does onboarding usually take for a mid-size company? Asking for planning purposes.
- `Medium` → `Low` (0.36): We're a 12-person startup. Maybe next quarter we'll need something like this. What's the cheapest plan?
- `Medium` → `High` (0.48): How does billing work for seats added mid-month? Curious before we look deeper.
- `Medium` → `High` (0.58): Hi! Could someone explain what a 'seat' means in your pricing? Trying to estimate costs.
- `Medium` → `Low` (0.37): We're running an RFP process starting in the spring. Could you send general product information?
- `Low` → `Medium` (0.39): GET RICH QUICK!!! Forex signals 95% win rate, join our telegram
- `Low` → `Medium` (0.44): Hi I'm 14 and doing a school project on AI, what is machine learning?
- `Low` → `Medium` (0.38): STOP
- `Low` → `Medium` (0.43): Could you recommend a good laptop for gaming?
- `Low` → `Medium` (0.36): ok
- `Low` → `Medium` (0.40): Testing the form, ignore this.
- `Low` → `High` (0.35): I want to cancel my gym membership.
- `Low` → `Medium` (0.37): Do you buy used cars?
- `Low` → `Medium` (0.43): Sending you a link to my YouTube channel, please subscribe!
- `Low` → `Medium` (0.43): asdf asdf asdf
- `Low` → `Medium` (0.40): We offer cheap cloud hosting, migrate today and get 3 months free!
