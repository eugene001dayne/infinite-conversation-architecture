"""
Infinite Conversation Architecture
Pre-Fetch Benchmark — Synthetic Conversation Generator

Author: Eugene Mawuli Attigah
Project: infinite-conversation-architecture
License: CC BY 4.0

Generates synthetic conversations with planted facts for the
Needle-in-a-Haystack benchmark design.

Each conversation has:
- A specific verifiable fact planted at a known turn
- A retrieval question at a known later turn that requires the fact
- Ground truth: the node_id of the turn containing the planted fact

Usage:
    python generate_conversations.py --count 100 --turns 50 --output conversations.json
"""

import json
import uuid
import random
import argparse
from datetime import datetime, timedelta

PLANTED_FACTS = [
    ("My dog's name is {name}.", "What should I name my new puppy? My current dog might be jealous."),
    ("I work at a company called {name}.", "Can you draft an email signature for me using my company name?"),
    ("My favourite programming language is {name}.", "I need to pick a language for a new project. What do you know about my preferences?"),
    ("My birthday is on the {day}th of {month}.", "What would be a good gift theme for someone with my birthday coming up?"),
    ("I am building a tool called {name}.", "Can you write a short tagline for the tool I am working on?"),
    ("My manager's name is {name}.", "I need to write an update for my manager. Can you help?"),
    ("I live in {name}.", "What is the time zone I should use for a meeting with someone in New York?"),
    ("My team has {count} people.", "How should I structure a standup for my team size?"),
    ("The project deadline is {month} {day}.", "How many weeks do I have left if my deadline is what I mentioned earlier?"),
    ("I am learning {name} in my spare time.", "Can you recommend a next step for what I am studying?"),
]

FILLER_NAMES = [
    "Atlas", "Zephyr", "Cobalt", "Iris", "Borealis", "Orion", "Lyra",
    "Quantum Labs", "Vertex AI", "Cascade Systems", "Meridian Tech",
    "Iron-Thread", "NeuralFlow", "GraphMind", "PulseAI",
    "Python", "Rust", "Elixir", "Julia", "Zig",
    "Sarah", "Marcus", "Priya", "James", "Amara",
    "Accra", "Lagos", "Nairobi", "Berlin", "Singapore"
]

FILLER_TOPICS = [
    ("How does exponential backoff work?", "It is a retry strategy where you wait progressively longer between retries."),
    ("What is the difference between a process and a thread?", "A process has its own memory space. Threads share memory within a process."),
    ("Can you explain what a webhook is?", "A webhook is an HTTP callback that fires when a specific event happens."),
    ("What is the best way to structure a REST API?", "Use nouns for resources, HTTP verbs for actions, and return consistent response shapes."),
    ("How do I write a good commit message?", "Start with a short imperative summary, then add context in the body if needed."),
    ("What is the CAP theorem?", "It states that a distributed system can guarantee at most two of: consistency, availability, partition tolerance."),
    ("How does HTTPS work?", "It uses TLS to encrypt traffic between client and server using asymmetric key exchange."),
    ("What is the difference between SQL and NoSQL?", "SQL databases are relational and schema-based. NoSQL trades structure for flexibility and scale."),
    ("Can you explain dependency injection?", "It is a pattern where dependencies are passed in rather than created inside a class."),
    ("What is memoisation?", "It is an optimisation where you cache the results of expensive function calls."),
    ("How do I handle errors in async Python?", "Use try/except inside async functions and handle CancelledError separately."),
    ("What is a binary search tree?", "A tree where each node's left children are smaller and right children are larger."),
    ("How does garbage collection work?", "It tracks object references and frees memory when objects have no more references."),
    ("What is the difference between TCP and UDP?", "TCP guarantees delivery and order. UDP is faster but has no guarantees."),
    ("Can you explain what a closure is?", "A function that captures variables from its enclosing scope even after that scope has closed."),
]


def random_name():
    return random.choice(FILLER_NAMES)


def random_day():
    return str(random.randint(1, 28))


def random_month():
    return random.choice(["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"])


def random_count():
    return str(random.randint(3, 20))


def fill_template(template):
    return template.format(
        name=random_name(),
        day=random_day(),
        month=random_month(),
        count=random_count()
    )


def generate_conversation(conv_id, total_turns, plant_at_turn):
    """
    Generate a synthetic conversation with a planted fact.

    Returns a dict with:
        conversation_id: str
        turns: list of {turn_number, speaker, text}
        planted_fact_turn: int
        planted_fact_text: str
        retrieval_question_turn: int
        retrieval_question_text: str
        ground_truth_note: str
    """
    fact_template, question_template = random.choice(PLANTED_FACTS)
    planted_fact = fill_template(fact_template)
    retrieval_question = question_template

    turns = []
    timestamp = datetime(2026, 3, 1, 9, 0, 0)

    for turn_num in range(1, total_turns + 1):
        timestamp += timedelta(minutes=random.randint(1, 3))

        if turn_num == plant_at_turn:
            user_text = f"By the way — {planted_fact} Anyway, back to what we were discussing."
            turns.append({"turn_number": turn_num, "speaker": "user", "text": user_text, "timestamp": timestamp.isoformat()})
            turns.append({"turn_number": turn_num, "speaker": "assistant", "text": "Got it, thanks for sharing that. Continuing on — " + random.choice(FILLER_TOPICS)[1], "timestamp": (timestamp + timedelta(seconds=3)).isoformat()})

        elif turn_num == total_turns:
            turns.append({"turn_number": turn_num, "speaker": "user", "text": retrieval_question, "timestamp": timestamp.isoformat()})

        else:
            topic_q, topic_a = random.choice(FILLER_TOPICS)
            turns.append({"turn_number": turn_num, "speaker": "user", "text": topic_q, "timestamp": timestamp.isoformat()})
            turns.append({"turn_number": turn_num, "speaker": "assistant", "text": topic_a, "timestamp": (timestamp + timedelta(seconds=3)).isoformat()})

    return {
        "conversation_id": conv_id,
        "total_turns": total_turns,
        "planted_fact_turn": plant_at_turn,
        "planted_fact_text": planted_fact,
        "retrieval_question_turn": total_turns,
        "retrieval_question_text": retrieval_question,
        "ground_truth_note": f"The node at turn {plant_at_turn} (user message) must appear in top-K retrieved nodes when the model processes turn {total_turns}.",
        "turns": turns
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conversations for the ICA pre-fetch benchmark")
    parser.add_argument("--count", type=int, default=100, help="Number of conversations to generate")
    parser.add_argument("--turns", type=int, default=50, help="Turns per conversation")
    parser.add_argument("--plant-at", type=int, default=10, help="Turn number to plant the fact (default: 10)")
    parser.add_argument("--output", type=str, default="conversations.json", help="Output file path")
    args = parser.parse_args()

    print(f"Generating {args.count} conversations of {args.turns} turns each...")
    print(f"Fact planted at turn {args.plant_at}, retrieval question at turn {args.turns}")

    conversations = []
    for i in range(args.count):
        conv_id = str(uuid.uuid4())
        conv = generate_conversation(conv_id, args.turns, args.plant_at)
        conversations.append(conv)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.count}...")

    output = {
        "benchmark_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "conversation_count": args.count,
            "turns_per_conversation": args.turns,
            "planted_fact_turn": args.plant_at,
            "retrieval_question_turn": args.turns,
            "hot_window_size": 25,
            "description": "Needle-in-a-Haystack benchmark dataset for ICA pre-fetch evaluation"
        },
        "conversations": conversations
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Saved {args.count} conversations to {args.output}")
    print(f"Ground truth: node at turn {args.plant_at} must be in top-K retrieved for each conversation.")


if __name__ == "__main__":
    main()
