import os

from dotenv import load_dotenv
from google import genai


# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Create the Gemini client
client = genai.Client(api_key=api_key)


def generate_answer(question, context):
    """
    Generate an answer using the user's question and retrieved code context.
    """

    prompt = f"""
You are an AI assistant that answers questions about a codebase.

Use only the provided code context to answer the question.
Do not invent functions, classes, files, or behavior that are not present
in the provided context.

If the context does not contain enough information to answer the question,
say that the answer cannot be determined from the retrieved code.

CODE CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text