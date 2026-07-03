from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone_setup import initialize_pinecone

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# 2. Connect to Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llmmodel = genai.GenerativeModel("gemini-3.1-flash-lite")

# 3. Function to retrieve context from Pinecone
def query_index(index, model, query_text, top_k=5):
    query_vector = model.encode([query_text])[0].tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result

# 4. Function to generate response with Gemini
def generate_response(results, query):
    prompt = f"""    
    You are an expert in groundwater assessment and management. Your role is to provide accurate, data-driven responses based strictly on the given context.

    Rules:
    Provide answers that are clear, concise, and factual.
    Always include units, figures, and values exactly as they appear in the context (no modifications or approximations).

    Style:
    Use a professional and technical tone.
    Structure responses logically for readability.
    Avoid speculation or assumptions.

    Context:
    {results}

    Query: {query}
    Answer:
    """
    response = llmmodel.generate_content(prompt)
    return response.text

# 5. Example usage
API_KEY = os.getenv("PINECONE_API_KEY")
index = initialize_pinecone(API_KEY)
model = SentenceTransformer('all-mpnet-base-v2')

query_text = input("Enter your query: ")
results = query_index(index, model, query_text)

answer = generate_response(results,query_text)
print(answer)
