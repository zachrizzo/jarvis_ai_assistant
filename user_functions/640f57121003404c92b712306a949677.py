import random

# List of jokes
jokes = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "Why was the math book sad? Because it had too many problems.",
    "What do you call a fake noodle? An impasta.",
    "Why did the scarecrow win an award? Because he was outstanding in his field.",
    "Why don't eggs tell jokes? They'd crack each other up."
]

def tell_joke():
    # Choose a random joke from the list
    joke = random.choice(jokes)
    print("Here's your joke: ")
    print(joke)

# Call the function to tell a joke
tell_joke()