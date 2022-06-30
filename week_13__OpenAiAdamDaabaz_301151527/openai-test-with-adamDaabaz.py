import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
"""completion"""
start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

response = openai.Completion.create(
  engine="davinci",
  prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: What is the color of",
  temperature=0.9,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.4,
  stop=["\n", " Human:", " AI:"]
)

print(response)


"""Q&A"""


openai.api_key = os.getenv("OPENAI_API_KEY")
"""
start_sequence = "\nA:"
restart_sequence = "\n\nQ: "
"""
response = openai.Completion.create(
  engine="davinci",
  prompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ:What is 1+1? 2+4?\nA:2\n\nQ:What is the color of the sky?\nA:",
  temperature=0,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)

print(response)

""" Summarize for the 2nd grader"""

response = openai.Completion.create(
  engine="davinci",
  prompt="My second grader asked me what this passage means:\n\"\"\"\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter. When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows, and is on average the third-brightest natural object in the night sky after the Moon and Venus.\n\"\"\"\n I rephrased it for him, in plain language a second grader can understand:\n\"\"\"\"",
  temperature=0.5,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0.2,
  presence_penalty=0,
  stop=["\"\"\""]
)

print(response)