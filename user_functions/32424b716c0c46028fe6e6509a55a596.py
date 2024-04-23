import random

# Define a list of jokes with their corresponding punchlines
jokes = [
    {"setup": "Why don't scientists trust atoms?", "punchline": "Because they make up everything."},
    {"setup": "Why don't eggs tell jokes?", "punchline": "They'd crack each other up."},
    {"setup": "Why did the tomato turn red?", "punchline": "Because it saw the salad dressing!"},
    # Add more jokes here...
]

def get_random_joke():
    """Return a random joke from memory."""
    if not jokes:
        return None
    return random.choice(jokes)

# Example usage
random_joke = get_random_joke()
if random_joke is not None:
    print(f"{random_joke['setup']} {random_joke['punchline']}")
else:
    print("No jokes available.")