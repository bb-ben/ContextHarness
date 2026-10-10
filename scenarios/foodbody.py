"""FoodBody-style scenarios: short histories with one known preference change.

Each scenario is a few dated sessions in which a person's preferences change after a
life event, written in HorizonBench's item format so every harness command works on
them: `--dataset scenarios/foodbody.jsonl`. They are small (about 8 events), so a
full run costs cents, and their labels are written to match the conversations, which
generated benchmark items do not always do.

What each scenario checks:
- the old value is replaced, not merged with the new one, and the change is noted;
- what stays the same (cooking time, units, format) survives the change;
- the reply to the last message follows the new values.

Run `python scenarios/foodbody.py` to rebuild scenarios/foodbody.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

T = "T09:00:00.000001"


def _conversation(sessions: list[tuple[str, str | None, list[tuple[str, str]]]]) -> str:
    lines = ["Conversation History:"]
    for date, scenario, turns in sessions:
        lines.append(f"Date: {date}{T}")
        if scenario:
            lines.append(f"Scenario: {scenario}")
        lines += [f"{role}: {text}" for role, text in turns]
    return "\n".join(lines) + "\n"


def _item(item_id, user, domain, sessions, options, correct, trap, original, current, event_date, event_name, category):
    changed = {k: {"original": original[k], "current": current[k]} for k in current if original[k] != current[k]}
    return {
        "id": item_id,
        "generator": "foodbody-scenarios",
        "user_id": f"foodbody-scenarios/{user}",
        "conversation": _conversation(sessions),
        "correct_letter": correct,
        "options": json.dumps([{"letter": L, "value": v, "option": o} for L, v, o in options]),
        "has_evolved": "true",
        "preference_domain": domain,
        "distractor_letter": trap,
        "preference_evolution": json.dumps({
            "original_attributes": original,
            "current_attributes": current,
            "changed_attributes": changed,
            "evolution_history": [{"date": f"{event_date}{T}", "event_name": event_name, "event_category": category,
                                   "trigger": "evolution_event",
                                   "attribute_changes": {k: {"from": v["original"], "to": v["current"]}
                                                         for k, v in changed.items()}}],
        }),
    }


U, A = "User", "Assistant"

DIAGNOSIS = _item(
    "Priya Raman_foodbody_diagnosis_meal_planning",
    "priya", "meal_planning_style",
    [
        ("2025-09-01", "A grad student wants a quick dinner idea.", [
            (U, "ok I need dinner ideas, nothing fancy. just throw a few at me, I hate rigid meal plans lol"),
            (A, "Fried rice with frozen veg, a big quesadilla, or pasta with garlic and spinach. All under 20 minutes."),
            (U, "perfect. and please no calorie stuff, numbers stress me out"),
            (A, "Noted: ideas only, no numbers."),
        ]),
        ("2025-09-05", "Snacks for a late study night.", [
            (U, "study night snacks?? make it fun"),
            (A, "Popcorn with chili-lime, apple slices with peanut butter, or a 'snack board' of whatever's in the fridge."),
            (U, "haha love the vibe, keep it like that"),
        ]),
        ("2025-09-10", "Using leftovers fast.", [
            (U, "got a ton of leftover rice. something in under 20 min?"),
            (A, "Egg fried rice: scramble two eggs, add rice, soy sauce, frozen peas. Ten minutes."),
        ]),
        ("2025-09-14", "Weekly groceries.", [
            (U, "grocery list for the week but keep it loose, I'll wing the actual meals"),
            (A, "Eggs, rice, tortillas, cheese, frozen veg, spinach, pasta, canned beans, fruit. Mix and match."),
        ]),
        ("2025-09-20", "Back from a doctor's appointment.", [
            (U, "just got back from the doctor. type 2 diabetes. kind of freaking out tbh. what does this even mean for food"),
            (A, "That's a lot to take in. Food still matters a lot and is very manageable: steady carbs, more fiber and "
                "protein. Your doctor or a dietitian can set targets with you."),
        ]),
        ("2025-09-24", "Asks for a real plan.", [
            (U, "ok I saw a dietitian. can you make me an actual plan for the week? breakfast lunch dinner. "
                "and I need carb counts per meal now, she wants me under 45g a meal"),
            (A, "Here's a Monday-to-Sunday plan, each meal with its carbs, all under 45 g and under 20 minutes."),
            (U, "this is so much better. I actually need the structure now, winging it isn't working anymore"),
        ]),
        ("2025-09-29", "A quick dinner swap.", [
            (U, "swap tuesday dinner for chili? how many carbs? still need it under 20 min"),
            (A, "Quick turkey-bean chili: about 38 g carbs per bowl, 18 minutes."),
            (U, "the per-meal carb numbers are exactly what I need. keep doing that"),
        ]),
        ("2025-10-03", "Next week's plan.", [
            (U, "lay out next week the same way pls. days, meals, carbs. still keep it fun though"),
            (A, "Week 2 plan, same layout, with a 'wild card' dinner on Friday."),
        ]),
        ("2025-10-08", "Tired after a long day.", [
            (U, "ugh long day. what should I make for dinner tonight?"),
        ]),
    ],
    [
        ("A", "loose_ideas", "Long day, I feel you! Quick ideas: egg fried rice, a cheesy quesadilla, or garlic spinach "
                             "pasta. Grab whatever sounds good and wing it, no numbers needed tonight."),
        ("B", "structured_with_carbs", "Long day, I feel you! Tonight's plan: egg fried rice with extra veg and an egg, "
                                       "about 40 g carbs, ready in 15 minutes. Want me to slot it into this week's plan?"),
        ("C", "detailed_nutrition_lecture", "Managing diabetes means balancing carbohydrates, fiber, protein and fat at "
                                            "every meal. Let's review glycemic index first, then build a dinner from "
                                            "principles."),
        ("D", "slow_gourmet", "Treat yourself: a slow-braised lentil stew with homemade flatbread takes about an hour "
                              "and a half, but it's worth it on a hard day."),
    ],
    correct="B", trap="A",
    original={"planning_style": "loose_ideas", "nutrition_detail": "no_numbers", "cooking_time": "under_20_min",
              "tone": "playful"},
    current={"planning_style": "structured_plan", "nutrition_detail": "carb_counts_per_meal",
             "cooking_time": "under_20_min", "tone": "playful"},
    event_date="2025-09-20", event_name="Diagnosed with type 2 diabetes", category="health",
)

INJURY = _item(
    "Tom Okafor_foodbody_injury_coaching",
    "tom", "coaching_style",
    [
        ("2025-08-04", "A runner training for a marathon.", [
            (U, "Training for a sub-3:30 marathon. Be brutal with me. Weekly plan in a table, km please."),
            (A, "| Day | Session | km |\n|---|---|---|\n| Tue | Intervals 6x1k | 12 |\n| Thu | Tempo | 14 |\n| Sun | Long run | 28 |"),
        ]),
        ("2025-08-09", "Feedback on a missed workout.", [
            (U, "Skipped Thursday's tempo, felt tired."),
            (A, "Tired isn't injured. Make it up Friday, no excuses."),
            (U, "Yes. Exactly that. Push harder, don't coddle me."),
        ]),
        ("2025-08-15", "Fueling a long run.", [
            (U, "How should I fuel a 32 km long run? Table please."),
            (A, "| km | Fuel |\n|---|---|\n| 0 | 60 g carb breakfast |\n| 10, 20, 28 | gel, 25 g carb |\n| after | 1.2 g/kg carb + protein |"),
        ]),
        ("2025-08-22", "Race-week meals.", [
            (U, "Race week meal plan. Performance only, I'll eat whatever wins."),
            (A, "Carb-load days 3-1: 8-10 g/kg carbs, low fiber, familiar foods. Table attached."),
        ]),
        ("2025-08-30", "An injury.", [
            (U, "Stress fracture in my foot. 8 weeks no running. Doctor says nutrition matters for healing. Gutted."),
            (A, "I'm sorry, that's a hard blow. Calcium, vitamin D, protein and enough total energy all support bone "
                "healing."),
        ]),
        ("2025-09-03", "How they want to be coached now.", [
            (U, "Honestly I need you to go easier on me right now. The last plan made me feel like a failure. "
                "Can we focus on healing, not performance?"),
            (A, "Of course. Healing is the job now, and you're doing it."),
        ]),
        ("2025-09-08", "Bone-healing meals.", [
            (U, "Meal ideas for bone healing? Table please, still in metric."),
            (A, "| Meal | Calcium | Protein |\n|---|---|---|\n| Greek yogurt + oats | 300 mg | 20 g |\n| Salmon, greens | 250 mg | 35 g |"),
            (U, "That was kind and useful. Helps way more than the drill-sergeant thing did."),
        ]),
        ("2025-09-14", "A setback.", [
            (U, "Ate pizza three nights in a row. Feel like I'm undoing my recovery. What now?"),
        ]),
    ],
    [
        ("A", "blunt_performance", "Three nights of pizza is three nights wasted. No excuses: lock in the plan, "
                                   "hit your protein, and get back to race shape. Table below, km as always."),
        ("B", "gentle_recovery", "Three nights of pizza doesn't undo your recovery. Healing is steady, not perfect. "
                                 "Here's a gentle table for the next few days, km as always, focused on calcium."),
        ("C", "vague_reassurance", "Don't worry about it at all! Everyone has off days. Just do whatever feels right "
                                   "and listen to your body, there's no need for any plan right now."),
        ("D", "clinical_essay", "Bone remodeling depends on calcium, vitamin D, protein and energy availability. "
                                "Let me explain osteoblast activity before we discuss what to eat."),
    ],
    correct="B", trap="A",
    original={"challenge_level": "push_hard", "focus": "performance", "format": "tables_metric"},
    current={"challenge_level": "gentle_supportive", "focus": "recovery", "format": "tables_metric"},
    event_date="2025-08-30", event_name="Stress fracture, eight weeks without running", category="health",
)

# The last two scenarios end on a neutral message, so the reply can't take its cue from
# the message itself: without memory there is nothing to say "short", "vegetarian" or
# "no numbers".

NEW_PARENT = _item(
    "Dani Morales_foodbody_new_parent",
    "dani", "cooking_help_style",
    [
        ("2025-06-02", "A weekend cooking project.", [
            (U, "Weekend project: fresh pasta from scratch. Explain the why behind every step, I love the science."),
            (A, "Gluten networks form as you knead: the proteins align, which is why resting the dough relaxes it..."),
            (U, "Love the detail. More of this always."),
        ]),
        ("2025-06-08", "A long Sunday cook.", [
            (U, "Vegetarian ramen broth with real depth? Go deep on technique, I've got all Sunday. "
                "And make it spicy, I put chili crisp on everything."),
            (A, "Roast kombu and shiitake for umami, char the aromatics, then a two-hour simmer... finish with chili crisp."),
        ]),
        ("2025-06-15", "Troubleshooting bread.", [
            (U, "Why is my sourdough dense? Give me the full rundown."),
            (A, "Usually underproofing: the starter, the temperature and the flour's protein all play a part. In detail..."),
        ]),
        ("2025-06-22", "Cooking for friends.", [
            (U, "Planning a six-hour vegetarian mole for friends. Walk me through it properly."),
            (A, "Toast the chiles, then the seeds and spices separately, because each burns at a different point..."),
            (U, "Honestly this is the best part of my week."),
        ]),
        ("2025-07-01", "Big news.", [
            (U, "Baby arrived early!! Total chaos, sleeping in 90-minute chunks."),
            (A, "Congratulations! Look after yourselves; food can be simple for a while."),
        ]),
        ("2025-07-10", "Dinner with one free hand.", [
            (U, "Need a dinner I can make one-handed. Keep it SHORT, like 3 lines. I read your answers at 3am."),
            (A, "Sheet-pan chickpeas, peppers, harissa. 25 min at 220°C. Eat with pita."),
        ]),
        ("2025-07-16", "Batch cooking.", [
            (U, "Freezer meals I can batch on one day? One pan if possible. Short list, no explanations."),
            (A, "1. Black bean chili\n2. Spicy lentil dal\n3. Veggie enchilada bake"),
            (U, "Perfect. Short is all I can handle now."),
        ]),
        ("2025-07-24", "Another quick one.", [
            (U, "Another one-pan thing? Still vegetarian obviously, and spicy, my taste buds are the only thing awake."),
            (A, "One-pan gochujang tofu and rice. 30 minutes."),
        ]),
        ("2025-07-30", None, [
            (U, "Any ideas for dinner this week?"),
        ]),
    ],
    [
        ("A", "weekend_project", "How about a weekend project: hand-rolled vegetarian dumplings. Here's why the "
                                 "dough rests, how pleating works, and the science of a crisp bottom."),
        ("B", "brief_one_pan", "Three quick ones: spicy black bean sheet-pan tacos, harissa chickpea tray bake, "
                               "chili-crisp fried rice. All one pan, all freeze well."),
        ("C", "brief_with_meat", "Three quick ones: spicy chicken sheet-pan fajitas, sausage and pepper tray bake, "
                                 "beef fried rice. All one pan, all freeze well."),
        ("D", "mild_weekly_plan", "Here's a balanced weekly plan: Monday mild vegetable soup, Tuesday pasta "
                                  "primavera, Wednesday cheese omelette, then leftovers to finish the week."),
    ],
    correct="B", trap="A",
    original={"answer_length": "detailed_explanations", "cooking_style": "elaborate_weekend_projects",
              "diet": "vegetarian", "spice": "loves_spicy"},
    current={"answer_length": "very_brief", "cooking_style": "one_pan_freezer_friendly",
             "diet": "vegetarian", "spice": "loves_spicy"},
    event_date="2025-07-01", event_name="Became a parent", category="family",
)

BURNOUT = _item(
    "Lena Fischer_foodbody_burnout",
    "lena", "nutrition_tracking_style",
    [
        ("2025-03-03", "Setting up nutrition goals.", [
            (U, "Log this: 140 g protein a day. I want exact grams for every meal you plan. Mediterranean style, "
                "and I'm cooking for me and my partner."),
            (A, "Got it: 140 g protein, Mediterranean, meals for two, with grams per meal."),
        ]),
        ("2025-03-09", "A weekly review.", [
            (U, "Weekly report please: my fats were off by 12%. Break the numbers down by day."),
            (A, "Mon 82 g, Tue 95 g, Wed 71 g... the over-runs were olive oil and feta on Tuesday and Friday."),
        ]),
        ("2025-03-16", "Planning dinners.", [
            (U, "Plan 7 dinners for two, Mediterranean, exact macros per serving."),
            (A, "1. Lemon chicken with farro: 42 g protein, 38 g carbs, 14 g fat per serving..."),
            (U, "The per-serving grams are perfect."),
        ]),
        ("2025-03-23", "Another report.", [
            (U, "Numbers again: what was my weekly protein average?"),
            (A, "131 g a day, 9 g under target; Thursday was the low day."),
        ]),
        ("2025-04-02", "Time off work.", [
            (U, "I've been signed off work with burnout. My therapist thinks the constant tracking is feeding my anxiety."),
            (A, "I'm sorry. Rest comes first, and we can change how we do food whenever you want."),
        ]),
        ("2025-04-08", "A decision about tracking.", [
            (U, "I'm stopping tracking. Please don't show me grams or calories anymore, seeing them sends me spiralling."),
            (A, "Understood. No numbers from now on."),
        ]),
        ("2025-04-15", "A comforting dinner.", [
            (U, "Dinner for two tonight, something Mediterranean and comforting? No numbers."),
            (A, "A slow-baked white bean and tomato stew with crusty bread and a big green salad."),
            (U, "Thank you for not putting numbers in. That felt calm."),
        ]),
        ("2025-04-22", "How to check in.", [
            (U, "Instead of the weekly reports, could you just check in with me gently about how eating's going?"),
            (A, "Of course. I'll ask how things feel, not how they measure."),
        ]),
        ("2025-04-28", None, [
            (U, "Planning groceries for the week, can you help?"),
        ]),
    ],
    [
        ("A", "macro_targets", "Sure! For 140 g protein a day for two: 1.2 kg chicken, 1 kg Greek yogurt, 12 eggs, "
                               "500 g lentils. I'll track the weekly totals and report back on Sunday."),
        ("B", "gentle_no_numbers", "Happy to. For the two of you: chickpeas, lemons, feta, tomatoes, greens, crusty "
                                   "bread, olive oil, fish for one night. How has eating felt this week?"),
        ("C", "generic_single", "Sure. Here's a simple list for one: pasta, a jar of sauce, frozen pizza, cereal, "
                                "milk, bananas, sandwich bread and cheese. Anything else you need?"),
        ("D", "calorie_budget", "Happy to. A balanced Mediterranean week for two at about 1,900 kcal each: "
                                "chickpeas, salmon, feta, greens. I'll keep a running calorie count."),
    ],
    correct="B", trap="A",
    original={"nutrition_numbers": "exact_macros", "check_in_style": "weekly_data_reports",
              "cuisine": "mediterranean", "household": "cooking_for_two"},
    current={"nutrition_numbers": "no_numbers", "check_in_style": "gentle_check_ins",
             "cuisine": "mediterranean", "household": "cooking_for_two"},
    event_date="2025-04-02", event_name="Signed off work with burnout", category="health",
)

SCENARIOS = [DIAGNOSIS, INJURY, NEW_PARENT, BURNOUT]

if __name__ == "__main__":
    out = Path(__file__).with_suffix(".jsonl")
    out.write_text("".join(json.dumps(s) + "\n" for s in SCENARIOS))
    print(f"Wrote {len(SCENARIOS)} scenarios to {out}")
