#!/usr/bin/env python3
"""Generate the prompt dataset: prompts/{factual,reasoning,role_based,instruction,creative}.json

Design (controlled variables):
  * 5 categories x N topics (default 2).
  * 3 length classes. SHORT is the bare prompt (<100 tokens). MEDIUM (~500 tokens) and
    LONG (>1,000 tokens) prepend the SAME neutral background block, so only length changes.
  * Keyword tests run at SHORT length: the same question with a prefix such as
    "Think step by step." / "You are an expert ..." / "Answer concisely in one sentence." /
    "Explain in detail and list five strategies."

Usage:
  python make_prompts.py                         # 2 topics per category -> 70 prompts
  python make_prompts.py --topics-per-category 1 # 35 prompts (fastest)
  python make_prompts.py --topics-per-category 3 # 105 prompts
"""
import argparse
import json
import random
from pathlib import Path

# Each topic: text = the base question, role = expert persona used by the role keyword test.
CATEGORIES = {
    "factual": [
        {"topic": "capital_france", "text": "What is the capital of France?", "role": "geography teacher"},
        {"topic": "largest_planet", "text": "What is the largest planet in our solar system?", "role": "astronomer"},
        {"topic": "romeo_juliet", "text": "Who wrote the play Romeo and Juliet?", "role": "literature teacher"},
    ],
    "reasoning": [
        {"topic": "train_distance",
         "text": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
         "role": "mathematics tutor"},
        {"topic": "apples",
         "text": "Sarah has 12 apples and gives 3 to John. John then gives 2 apples to Mark. How many apples does John have?",
         "role": "mathematics tutor"},
        {"topic": "muffins",
         "text": "A recipe needs 3 eggs for every 12 muffins. How many eggs are needed for 60 muffins?",
         "role": "mathematics tutor"},
    ],
    # role_based: the base prompt ALREADY contains the role, so the expert_role keyword is skipped.
    "role_based": [
        {"topic": "recursion", "text": "Explain recursion to a beginner.", "role": "computer science tutor"},
        {"topic": "photosynthesis", "text": "Explain photosynthesis to a beginner.", "role": "biology teacher"},
        {"topic": "supply_demand", "text": "Explain supply and demand to a beginner.", "role": "economics tutor"},
    ],
    "instruction": [
        {"topic": "recursion_list",
         "text": "Explain recursion, define it, provide an example, explain how it works, and list two advantages and two disadvantages.",
         "role": "computer science tutor"},
        {"topic": "hash_table",
         "text": "Explain hash tables, define them, provide an example, explain how lookups work, and list two advantages and two disadvantages.",
         "role": "computer science tutor"},
        {"topic": "vaccines",
         "text": "Explain how vaccines work, define the key terms, provide an example, describe the process, and list two benefits and two limitations.",
         "role": "biology teacher"},
    ],
    "creative": [
        {"topic": "ai_in_laptop",
         "text": "Write a short story about a programmer who discovers an AI living inside their laptop.",
         "role": "fiction writer"},
        {"topic": "mysterious_door",
         "text": "Write a short story about a girl who discovers a mysterious door in her apartment.",
         "role": "fiction writer"},
        {"topic": "lighthouse",
         "text": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
         "role": "fiction writer"},
    ],
}

FILES = {
    "factual": "factual.json",
    "reasoning": "reasoning.json",
    "role_based": "role_based.json",
    "instruction": "instruction.json",
    "creative": "creative.json",
}

# keyword name -> prefix template ({role} is filled per topic)
KEYWORDS = {
    "none": "",
    "think_step": "Think step by step.",
    "expert_role": "You are an expert {role}.",
    "concise": "Answer concisely in one sentence.",
    "detailed": "Explain in detail and list five strategies.",
}

# Rough word counts; Llama-style tokenizers give ~1.3-1.5 tokens/word on plain English.
# Run `python benchmark_runner.py --dry-run` to see the REAL token counts and adjust here.
TARGET_WORDS = {"medium": 330, "long": 800}

SENTENCES = [
    "The city library opened a new reading room last spring, and volunteers helped to organize the donated books by subject.",
    "Weather records show that the region receives most of its rainfall between October and March.",
    "A typical bicycle has two wheels, a frame, pedals, a chain, and a set of brakes that the rider controls by hand.",
    "Engineers often test a design in small steps so that mistakes are found before they become expensive.",
    "The museum keeps a collection of old maps that show how the coastline has changed over several centuries.",
    "Farmers in the valley plant wheat in the autumn and harvest it early the following summer.",
    "A balanced schedule includes time for work, rest, exercise, and meals shared with friends or family.",
    "The orchestra rehearsed twice a week, and the conductor made notes after every session.",
    "Public transport in the area includes buses, trams, and a small ferry that crosses the river every half hour.",
    "Many households keep a small notebook near the kitchen to track groceries and household repairs.",
    "The old bridge was repaired in stages, and each stage was inspected before traffic was allowed to resume.",
    "Students who review their notes soon after a lesson tend to remember the material for longer.",
    "A local bakery sells bread, pastries, and seasonal fruit tarts, and it often sells out before noon.",
    "Satellites orbiting the planet collect images that scientists use to study forests, oceans, and cities.",
    "The committee met on Thursday to discuss the budget, the schedule, and the list of upcoming events.",
    "Careful labeling of files and folders makes it much easier to find documents months later.",
    "The park contains a pond, two playgrounds, a walking trail, and a small garden maintained by neighbors.",
    "In many languages, the names of the days of the week come from ancient gods, planets, or celestial bodies.",
    "A good map includes a scale, a legend, and a clear indication of which direction is north.",
    "Researchers recorded the temperature each hour for a month and stored the readings in a shared spreadsheet.",
    "The ferry company announced a revised timetable that adds two extra departures on weekends.",
    "Pottery workshops teach students how to shape clay, glaze finished pieces, and fire them in a kiln.",
    "The mountain trail is marked with painted stones, and hikers are advised to carry water and a light jacket.",
    "After the festival, volunteers collected the bins, folded the tables, and swept the square before sunset.",
    "Home gardeners often start seeds indoors in early spring and move the seedlings outside once the frost has passed.",
    "The regional museum offers guided tours in the morning and self-guided visits throughout the afternoon.",
    "A reliable routine for backing up photographs includes copying them to two separate drives.",
    "Local radio stations broadcast traffic updates, weather forecasts, and community announcements every hour.",
    "The textile mill employed several hundred workers and produced fabric that was shipped to markets abroad.",
    "Tide tables list the expected times of high and low water and are used by sailors and fishing crews.",
]


def build_context(n_words: int, seed: int = 7) -> str:
    """Deterministic neutral filler; identical for every prompt of the same length class."""
    rng = random.Random(seed)
    pool = SENTENCES[:]
    out, words, i = [], 0, 1
    while words < n_words:
        rng.shuffle(pool)
        for s in pool:
            out.append(f"Note {i}. {s}")
            words += len(s.split()) + 2
            i += 1
            if words >= n_words:
                break
    return "\n".join(out)


def make_item(category, topic, length, keyword, contexts):
    base = topic["text"]
    if category == "role_based":
        base = f"You are an expert {topic['role']}. {base}"
    prefix = KEYWORDS[keyword].format(role=topic["role"])
    text = f"{prefix} {base}".strip()
    if length != "short":
        text = f"Background notes (for reference):\n{contexts[length]}\n\nTask: {text}"
    return {
        "id": f"{category}-{topic['topic']}-{length}-{keyword}",
        "category": category,
        "topic": topic["topic"],
        "length": length,
        "keyword": keyword,
        "prompt": text,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics-per-category", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--out", default="prompts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contexts = {k: build_context(v) for k, v in TARGET_WORDS.items()}

    total = 0
    for category, topics in CATEGORIES.items():
        items = []
        for topic in topics[: args.topics_per_category]:
            for keyword in KEYWORDS:
                if keyword == "expert_role" and category == "role_based":
                    continue  # base prompt already has the role
                items.append(make_item(category, topic, "short", keyword, contexts))
            for length in ("medium", "long"):
                items.append(make_item(category, topic, length, "none", contexts))
        (out / FILES[category]).write_text(json.dumps(items, indent=2))
        total += len(items)
        print(f"{FILES[category]:<18} {len(items):>3} prompts")
    print(f"Total: {total} prompts -> x models x trials = number of timed runs")


if __name__ == "__main__":
    main()
