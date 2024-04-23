import sqlite3

# Connect to the database (replace with your own database connection)
conn = sqlite3.connect("jokes.db")
cursor = conn.cursor()

# Retrieve a random joke from the database
cursor.execute("SELECT joke FROM jokes ORDER BY RANDOM() LIMIT 1")
result = cursor.fetchone()

if result:
    joke = result[0]
    print("Here's a joke for you:", joke)
else:
    print("No jokes found!")

# Close the database connection
conn.close()