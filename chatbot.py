import sys
import nltk
from nltk.chat.util import Chat, reflections

# Define the chatbot responses
pairs = [
    ["hi", ["Hello!", "Hi there!", "Hey!"]],
    ["hello", ["Hello!", "Hi there!", "Hey!"]],
    ["how are you", ["I'm doing well, thank you!", "I'm good, how about you?"]],
    ["what's your name", ["I'm a chatbot!", "You can call me Chatbot.", "I'm just a bot."]],
    ["bye", ["Goodbye!", "Bye!", "See you later!"]],
    ["goodbye", ["Goodbye!", "Bye!", "See you later!"]],
]

# Create the chatbot
chatbot = Chat(pairs, reflections)

# Run the chatbot
print("Welcome to the Chatbot. Type 'quit' to exit.")
while True:
    try:
        user_input = input("You: ")
    except EOFError:
        print("Chatbot: Goodbye!")
        sys.exit(0)

    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    else:
        response = chatbot.respond(user_input)
        print("Chatbot:", response)
