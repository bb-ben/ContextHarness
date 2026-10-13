"""Long FoodBody-style scenarios: about 35 events each, one change arriving gradually.

Written after the profile design was frozen, so they also serve as held-out tests.
Each history has unrelated requests mixed in, one lasting fact mentioned only once
early on (an allergy, an intolerance, a diet), a life event, and a change that shows
over many later requests rather than in one statement. Each ends on a neutral message.

Run `python scenarios/foodbody_long.py` to rebuild foodbody_long.jsonl and
foodbody_long_checklists.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from foodbody import A, U, _item  # noqa: E402


def ev(date: str, user: str, assistant: str):
    return (date, None, [(U, user), (A, assistant)])


RETIREMENT = _item(
    "Hannah Brooks_foodbody_long_retirement", "hannah", "home_cooking_style",
    [
        ev("2025-02-03", "Weeknight dinner for me and my two teenagers? Has to be on the table in 30 minutes.",
           "Pan-seared salmon with garlic green beans and rice: 25 minutes, serves three."),
        ev("2025-02-06", "Grocery list for the three of us this week. I don't mind splurging: the good olive oil, grass-fed beef.",
           "Grass-fed ground beef, wild salmon, aged parmesan, good olive oil, berries, sourdough, greens."),
        ev("2025-02-10", "Dessert for Leo's birthday? Nothing with peanuts, I'm allergic.",
           "A chocolate olive-oil cake, no nuts anywhere near it."),
        ev("2025-02-13", "What wine goes with short ribs?", "A Cabernet Sauvignon or a Syrah."),
        ev("2025-02-17", "Any tips for sleeping better? I keep waking at 4am.", "Keep a fixed wake time and a cool, dark room."),
        ev("2025-02-20", "Fast dinner, 30 minutes max, the kids are starving after practice.",
           "Steak tacos with quick-pickled onions: 20 minutes."),
        ev("2025-02-24", "Is a stand mixer worth it?", "If you bake weekly, yes."),
        ev("2025-02-28", "Splurge-worthy Saturday dinner for the three of us. I want to try dry-aged steak.",
           "Dry-aged ribeye, potato gratin, and a Bordeaux."),
        ev("2025-03-04", "How do I keep herbs fresh longer?", "Stems in water, loosely covered, in the fridge."),
        ev("2025-03-08", "Quick lunchbox ideas for two teenagers.", "Chicken wraps, pasta salad, cheese and crackers."),
        ev("2025-03-12", "Good stretches after a long day at a desk?", "Hip flexor lunges, doorway chest stretch."),
        ev("2025-03-17", "Wine pairing for a mushroom risotto?", "A Pinot Noir or an oaked Chardonnay."),
        ev("2025-03-21", "30-minute dinner again, it's chaos here.", "Sheet-pan chicken thighs and vegetables, 30 minutes."),
        ev("2025-03-26", "How long does cooked rice keep in the fridge?", "Up to four days, cooled quickly."),
        ev("2025-04-02", "Big week. Maya left for her exchange year, Leo moved into his dorm, and Friday was my last day at "
           "work: I retired. The house is so quiet.", "That's a lot of change at once. Congratulations on retiring."),
        ev("2025-04-06", "Cooking for just me now, and every recipe makes four. How do I scale down?",
           "Halve or quarter recipes, and freeze single portions."),
        ev("2025-04-10", "Any good podcasts about food history?", "Gastropod is a good start."),
        ev("2025-04-14", "I'm on a pension now and trying to watch what I spend. Cheaper cuts that still taste good?",
           "Chicken thighs, pork shoulder, beef chuck: all great braised."),
        ev("2025-04-18", "I've got the whole afternoon. Teach me to braise properly?",
           "Sear, add aromatics and liquid, cover, and cook low for three hours."),
        ev("2025-04-22", "What's the difference between baking soda and baking powder?",
           "Soda needs an acid; powder carries its own."),
        ev("2025-04-26", "Single-portion dinner that doesn't leave me a week of leftovers?", "A one-pan chicken thigh with lentils."),
        ev("2025-04-30", "Groceries are adding up. Budget meal plan for one, around $40 a week?",
           "Lentils, eggs, a whole chicken, rice, seasonal vegetables."),
        ev("2025-05-04", "Should I get a slow cooker?", "A Dutch oven does the same job if you have one."),
        ev("2025-05-08", "I want a weekend project. Bread from scratch, maybe? I finally have time.",
           "A simple overnight loaf is a good first project."),
        ev("2025-05-12", "How do I sharpen a knife?", "A whetstone at about 15 degrees, then hone."),
        ev("2025-05-16", "Dinner for one tonight. I'm happy to spend an hour cooking.", "Mushroom risotto for one: about 45 minutes."),
        ev("2025-05-20", "Ideas for a small herb garden on a balcony?", "Basil, thyme, rosemary and chives."),
        ev("2025-05-24", "Skip the fancy stuff. Beans, lentils, whatever's cheap and good.", "Black bean soup and red lentil dal."),
        ev("2025-05-28", "A friend is coming for coffee. A simple cake?", "A lemon drizzle loaf."),
        ev("2025-06-01", "Is it cheaper to buy whole chickens and break them down? I have time to learn.",
           "Yes, usually by about a third. Here's how to joint one."),
        ev("2025-06-05", "How do I store bread so it doesn't go stale?", "Cut side down on a board, or freeze sliced."),
        ev("2025-06-09", "A slow Sunday recipe for one? I like having something simmering.", "A small pot of ragu, two hours."),
        ev("2025-06-13", "Good walking routes to stay active? I walk every morning now.", "Try a loop with a hill in the middle."),
        ev("2025-06-17", "Frozen vegetables or fresh on a budget?", "Frozen is often cheaper and just as nutritious."),
        ev("2025-06-21", "Which spices are worth buying first?", "Cumin, smoked paprika, chili flakes, oregano."),
        ("2025-06-25", None, [(U, "What should I cook this week?")]),
    ],
    [
        ("A", "quick_family_premium", "A fast week for the three of you: 30-minute dinners with dry-aged steak, wild "
                                      "salmon and a wine pairing each night."),
        ("B", "budget_one_slow", "A relaxed week for one on a budget: a slow lentil braise, a whole roast chicken "
                                 "stretched into three meals, and Sunday bread."),
        ("C", "quick_one", "A quick week for one: microwave rice bowls, 10-minute pastas and frozen stir-fries, so "
                           "you're never stuck in the kitchen for long."),
        ("D", "peanut_budget", "A cheap and simple week for one: a peanut satay stir-fry, a crunchy peanut slaw, and "
                               "a slow-simmered Thai peanut curry."),
    ],
    correct="B", trap="A",
    original={"household": "family_of_three", "budget": "premium", "cooking_time": "quick_30_min",
              "allergy": "peanuts"},
    current={"household": "one_person", "budget": "budget_conscious", "cooking_time": "leisurely_projects",
             "allergy": "peanuts"},
    event_date="2025-04-02", event_name="Children left home and retired", category="life",
)

PREGNANCY = _item(
    "Ade Okonkwo_foodbody_long_pregnancy", "ade", "nutrition_coaching_style",
    [
        ev("2025-03-03", "Training for a half marathon in May. Set me macros: I want to lean out, 300-calorie deficit, "
           "140 g protein.", "Roughly 1,900 kcal, 140 g protein, 190 g carbs, 60 g fat."),
        ev("2025-03-06", "Didn't hit protein yesterday. Be blunt, I need someone to push me.",
           "You missed by 30 g. Fix it today: protein at every meal."),
        ev("2025-03-10", "Best running shoes for flat feet?", "Look at stability shoes; try them at a running store."),
        ev("2025-03-13", "High-protein breakfast that isn't yogurt? I'm lactose intolerant.",
           "Tofu scramble or eggs with black beans."),
        ev("2025-03-17", "How should I fuel a 16 km long run?", "A carb-rich breakfast, and a gel every 40 minutes."),
        ev("2025-03-20", "How do I stop my earbuds falling out when I run?", "Try ear hooks or a smaller tip size."),
        ev("2025-03-24", "Weekly check: I averaged 128 g protein. Push me.", "Not good enough. 140 means 140. Plan it."),
        ev("2025-03-28", "Meal plan for a cutting week, high protein.", "Chicken, lentils, eggs, tofu; 1,900 kcal a day."),
        ev("2025-04-01", "Is a sauna good after long runs?", "Fine if you rehydrate."),
        ev("2025-04-05", "Fast, high-protein dinner after my run?", "Salmon, quinoa and broccoli: 45 g protein."),
        ev("2025-04-09", "Gift idea for my sister's birthday?", "A cooking class for two."),
        ev("2025-04-13", "Am I eating too many carbs before tempo runs?", "No; carbs before tempo work help."),
        ev("2025-04-17", "Snack ideas that hit 30 g protein?", "A tuna wrap, or a protein shake made with water."),
        ev("2025-04-21", "How do I clean water bottles properly?", "Hot soapy water and a bottle brush, then air-dry."),
        ev("2025-04-25", "Keep me accountable this week. No excuses.", "Log every meal. I'll check your numbers Friday."),
        ev("2025-04-29", "Big news: I'm pregnant! Nine weeks. My doctor says no more hard training or calorie cutting "
           "for now.", "Congratulations! Let's follow your doctor's lead."),
        ev("2025-05-03", "Is sushi off the menu now?", "Raw fish is out for now; cooked rolls are fine."),
        ev("2025-05-06", "So nauseous in the mornings. Anything gentle I can keep down?", "Dry crackers, ginger tea, small meals."),
        ev("2025-05-10", "Comfortable shoes for walking? I'm swapping running for walks.", "Cushioned walking shoes."),
        ev("2025-05-13", "My doctor wants me eating enough, not counting. Can we drop the macro targets?",
           "Yes. No targets: just regular, balanced meals."),
        ev("2025-05-17", "Which cheeses are safe? Not that I eat much dairy anyway.",
           "Hard pasteurized cheeses; avoid soft unpasteurized ones."),
        ev("2025-05-20", "Tips for sleeping on my side?", "A pillow between the knees helps."),
        ev("2025-05-24", "Honestly I need encouragement more than pushing right now.", "Understood. You're doing well."),
        ev("2025-05-27", "Are deli sandwiches okay?", "Cold deli meats are best avoided unless heated until steaming."),
        ev("2025-05-31", "Iron-rich foods? My levels are a bit low.", "Lentils, spinach, beef, fortified cereals with vitamin C."),
        ev("2025-06-03", "What goes in a hospital bag? Way too early, I know.", "Comfortable clothes, snacks, chargers."),
        ev("2025-06-07", "Gentle dinner tonight, nothing heavy.", "Chicken and rice soup."),
        ev("2025-06-10", "Thanks for being kind about my missed walks this week. That helped.", "Rest is part of it."),
        ev("2025-06-14", "Names that work in both English and Yoruba?", "Tola, Ade, Femi, Kemi."),
        ev("2025-06-17", "Which fish is safe? I miss salmon.", "Cooked salmon is great; avoid high-mercury fish."),
        ev("2025-06-21", "Easy snacks for work? I get hungry every two hours now.", "Hummus and crackers, fruit, nuts."),
        ev("2025-06-24", "Is it normal to be this tired?", "Very common; mention it to your doctor."),
        ev("2025-06-28", "Folate-rich foods?", "Leafy greens, beans, fortified grains."),
        ev("2025-07-01", "Something comforting for dinner. Long day.", "A mild lentil and sweet potato stew."),
        ev("2025-07-05", "Is prenatal yoga worth trying?", "Many find it helpful; check with your doctor."),
        ("2025-07-08", None, [(U, "Can you plan my lunches for this week?")]),
    ],
    [
        ("A", "deficit_push", "Lunches to keep your 300-calorie deficit: tuna sashimi bowls at 45 g protein each. "
                              "No excuses on the macros this week."),
        ("B", "gentle_safe", "Gentle, filling lunches: lentil soup, chicken and rice bowls, and well-cooked salmon "
                             "with greens, all dairy-free and pregnancy-safe."),
        ("C", "unsafe_dairy", "Easy, satisfying lunches for the whole week: deli turkey subs, a brie and grape plate, "
                              "and smoked salmon bagels with plenty of cream cheese."),
        ("D", "macros_dairy", "High-protein lunches for the week: Greek yogurt bowls, cottage cheese wraps, and "
                              "whey shakes, so you still hit your targets."),
    ],
    correct="B", trap="A",
    original={"goal": "lean_out_deficit", "coaching_tone": "push_hard", "tracking": "macro_targets",
              "dairy": "lactose_free"},
    current={"goal": "nourish_pregnancy", "coaching_tone": "gentle_encouraging", "tracking": "no_counting",
             "dairy": "lactose_free"},
    event_date="2025-04-29", event_name="Pregnancy", category="health",
)

NIGHT_SHIFTS = _item(
    "Sam Rivera_foodbody_long_night_shifts", "sam", "food_help_style",
    [
        ev("2025-01-06", "Can you explain how protein combining works? I love the science, go deep.",
           "Amino acids, limiting amino acids, and why variety across a day matters more than per meal."),
        ev("2025-01-09", "Weekend batch cook: a big vegan chili. I want the full method and why each step matters.",
           "Bloom the spices in oil first, because their flavors are fat-soluble..."),
        ev("2025-01-13", "Good compression socks for 12-hour shifts?", "Graduated 15-20 mmHg socks."),
        ev("2025-01-16", "Why does soaking beans matter? Full explanation please.",
           "Soaking hydrates the beans evenly and leaches out some oligosaccharides..."),
        ev("2025-01-20", "Plan me a Sunday cooking project. I have four hours.", "Homemade tempeh bowls with three sauces."),
        ev("2025-01-23", "How do I get turmeric stains out of a cutting board?", "Baking soda paste, then sunlight."),
        ev("2025-01-27", "Explain B12 for vegans in detail.", "Why plants don't make it, absorption, and dosing."),
        ev("2025-01-30", "A detailed homemade seitan recipe, with the why.", "Vital wheat gluten, kneading for gluten networks..."),
        ev("2025-02-03", "Book recommendations about nursing history?", "Notes on Nursing, by Florence Nightingale."),
        ev("2025-02-06", "What's the science behind fermenting tempeh?", "Rhizopus mycelium binds the beans..."),
        ev("2025-02-10", "An elaborate vegan Sunday brunch?", "Chickpea-flour omelettes, cashew hollandaise, and waffles."),
        ev("2025-02-13", "Best way to clean a cast-iron pan?", "Hot water, a brush, dry, and a thin oil coat."),
        ev("2025-02-17", "Go deep on omega-3 sources for vegans.", "ALA conversion, algae oil, and how much you need."),
        ev("2025-02-20", "Moving to night shifts next week: 7pm to 7am, three nights a week.",
           "Big adjustment. Happy to help you plan around it."),
        ev("2025-02-24", "First night shift done. What should I eat at 3am? Keep it short.",
           "Something light: hummus wrap, fruit."),
        ev("2025-02-27", "Are blackout curtains worth it?", "Yes, for sleeping in the day."),
        ev("2025-03-03", "Our unit's microwave is broken half the time. Meals that don't need reheating?",
           "Grain salads, wraps, cold noodle bowls."),
        ev("2025-03-06", "Too long, I'm reading this half asleep. Just the list.", "- Soba salad\n- Chickpea wrap\n- Fruit"),
        ev("2025-03-10", "How do I nap before a shift without feeling groggy?", "Keep it to 20 minutes or a full 90."),
        ev("2025-03-13", "Bullet points please: portable vegan snacks for the night.",
           "- Roasted chickpeas\n- Trail mix\n- Rice cakes with peanut butter"),
        ev("2025-03-17", "When should I eat on a 7pm to 7am shift? Short answer.", "Meal at 6pm, light snack at 1am, small breakfast after."),
        ev("2025-03-20", "Gift for a coworker who's retiring?", "A nice notebook and a card from the team."),
        ev("2025-03-24", "Quick: a no-cook lunch I can grab before work?", "Bean salad in a jar."),
        ev("2025-03-27", "That was perfect. Short is what I need on these weeks.", "Got it."),
        ev("2025-03-31", "Is coffee at 4am okay?", "Better to stop caffeine about six hours before you sleep."),
        ev("2025-04-03", "Three lines: what helps with night-shift nausea?", "Small meals.\nGinger.\nAvoid greasy food."),
        ev("2025-04-07", "Something I can eat cold from a container at 2am?", "Peanut soba noodles."),
        ev("2025-04-10", "Sunglasses for driving home in the morning?", "Polarized, wraparound."),
        ev("2025-04-14", "List only: vegan proteins that keep without a fridge for 12 hours.",
           "- Roasted chickpeas\n- Nuts\n- Shelf-stable tofu"),
        ev("2025-04-17", "Short version please: breakfast after a night shift, before sleeping?", "Toast with nut butter."),
        ev("2025-04-21", "How do I explain my sleep schedule to my family?", "Share your shift calendar."),
        ev("2025-04-24", "Keep it brief: one weekend cook that covers three night shifts?", "A big tray of cold peanut noodles."),
        ev("2025-04-28", "Just bullets: what should I keep in my locker for food?", "- Nuts\n- Crackers\n- Oat bars"),
        ev("2025-05-01", "Running shoes for someone on their feet all day?", "Cushioned neutral trainers."),
        ev("2025-05-05", "Quick one: hydration on night shifts?", "A bottle per four hours; ease off before sleep."),
        ("2025-05-08", None, [(U, "Any ideas for meals I can take to work this week?")]),
    ],
    [
        ("A", "deep_science", "Let's start with the science: how circadian rhythm shifts digestion and why protein "
                              "timing matters, then a full Sunday seitan project to cook ahead."),
        ("B", "short_vegan_cold", "- Cold peanut soba with tofu\n- Chickpea salad wraps\n- Overnight oats with soy "
                                  "milk\nAll vegan, eaten cold, packable for a 7pm-7am shift."),
        ("C", "not_vegan", "Some easy ideas for the week: a chicken pasta salad, a tuna and cucumber wrap, and Greek "
                           "yogurt with granola, all simple to pack."),
        ("D", "needs_reheat", "Try a vegan lasagna, a red lentil soup, and a tofu and spinach curry. Make them on Sunday "
                              "and reheat each at work."),
    ],
    correct="B", trap="A",
    original={"answer_length": "detailed_science", "food_format": "elaborate_cooking", "diet": "vegan"},
    current={"answer_length": "short_bullets", "food_format": "portable_no_reheat", "diet": "vegan"},
    event_date="2025-02-20", event_name="Started night shifts", category="work",
)

SCENARIOS = [RETIREMENT, PREGNANCY, NIGHT_SHIFTS]

CHECKLISTS = {
    RETIREMENT["id"]: {
        "must_contain": ["Is allergic to peanuts", "Cooks for one person now (the children left home)", "Is retired",
                         "Is on a budget (fixed income) and wants cheap ingredients",
                         "Has time and enjoys slower cooking projects (braising, bread)"],
        "must_not_assert": ["Cooks for a family of three", "Likes splurging on premium ingredients",
                            "Needs dinners done in 30 minutes"],
        "reply_should": ["Sizes meals for one person", "Uses budget-friendly ingredients",
                         "Includes at least one slower, hands-on cooking project", "Contains no peanuts"],
    },
    PREGNANCY["id"]: {
        "must_contain": ["Is pregnant", "Is lactose intolerant",
                         "Avoids pregnancy-unsafe foods (raw fish, cold deli meat, soft unpasteurized cheese)",
                         "Wants gentle encouragement rather than pushing", "No calorie cutting or macro targets now",
                         "Has low iron"],
        "must_not_assert": ["Wants a calorie deficit to lean out", "Wants blunt pushing and accountability",
                            "Wants 140 g protein or other macro targets"],
        "reply_should": ["Contains no raw fish, cold deli meat or soft unpasteurized cheese", "Is dairy-free",
                         "Has no calorie deficit or macro targets", "Is gentle in tone"],
    },
    NIGHT_SHIFTS["id"]: {
        "must_contain": ["Is vegan", "Is a nurse working night shifts (7pm to 7am, three nights a week)",
                         "Wants short, bullet-point answers now",
                         "Needs portable meals that don't need reheating (unreliable microwave)",
                         "Eats around a night-shift schedule"],
        "must_not_assert": ["Wants long, detailed science explanations as the default",
                            "Wants elaborate weekend cooking projects"],
        "reply_should": ["Is short (a brief list)", "Suggests only vegan food",
                         "Suggests meals that can be eaten cold or without reheating",
                         "Fits a night shift (portable, timed for the shift)"],
    },
}

if __name__ == "__main__":
    out = Path(__file__).with_suffix(".jsonl")
    out.write_text("".join(json.dumps(s) + "\n" for s in SCENARIOS))
    checklists = Path(__file__).with_name("foodbody_long_checklists.json")
    checklists.write_text(json.dumps(CHECKLISTS, indent=2) + "\n")
    print(f"Wrote {len(SCENARIOS)} scenarios to {out} and their checklists to {checklists}")
