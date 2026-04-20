import os
import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from groq import Groq
from groq_api import groq_api

os.environ["GROQ_API_KEY"] = groq_api

def create_embedding(text_list):
    r =requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3", 
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    # Try Groq first
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content

    # If Groq fails, fall back to Ollama
    except Exception as e:
        print(f"Groq failed ({e}), falling back to Ollama...")
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        return r.json()["response"]

df = joblib.load("embeddings.joblib")
input_query = input("Ask a question: ")
question_embedding = create_embedding([input_query])[0]

# Find similarities between question_embedding and input_query
# print(np.vstack(df["embedding"].values)) #Converting Embeggings values into 2d numpy array
# print(np.vstack(df["embedding"]).shape) #Cosine similarity funtion onyl takes 2d array as input 
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()

top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]

prompt = f"""
You are an AI assistant that answers questions based on the provided video content.

Context:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

------------------------------------------------

User Question:
{input_query}

Instructions:
- Answer clearly and naturally (like a teacher explaining).
- Mention relevant video number(s) and timestamps.
- Use 2–4 most relevant timestamps (avoid too many).
- Convert timestamps to mm:ss format.
- Keep explanation helpful and easy to understand.
-Do not give invalid timestamps it should be correct the seconds should not excedd 60. If it exceedes you should add 1 minute to minutes and deduct 60 from the seconds 
-Do not write in the answer such things (579.24 seconds converted to mm:ss format is 9:51)

- If the question is unrelated, say:
  "This question is not related to the available video content."

Do NOT:
- Guess timestamps that are not present
- Mention anything like "not available" or "based on context"

Give a clean, human-like answer.
"""

response = inference(prompt)
print(response)
with open ("response.txt", "w") as f:
    f.write(response)