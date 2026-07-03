from neo4j import GraphDatabase
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# --- CONFIG ---
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# --- Your Knowledge Graph Schema ---
SCHEMA = """
We have a Neo4j knowledge graph with these entities:

Nodes and their properties:
(:Availability - command, non_command, poor_quality, total)
(:GroundWaterAvailability - command, non_command, poor_quality, total)
(:Aquifer - dynamic_gw, in_storage_gw, total, type)
(:Loss - command, non_command, poor_quality, total, et, evaporation, transpiration)
(:Rainfall - command, non_command, poor_quality, total)
(:AdditionalRecharge - floodProneArea, shallowArea, springDischarge, total)
(:Recharge - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall,sewage, streamRecharge, surface_irrigation, total, water_body)
(:BlockSummary - Hilly Area, critical, over_exploited, safe, semi_critical, salinity)
(:Area - type(non_recharge_worthy, recharge_worthy, total), commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
(:Draft - agriculture, domestic, industry, total)
(:Allocation - domestic, industry, total)
(:State - name, uuid)
(:StageOfExtraction - command, non_command, poor_quality, total)
(:FutureUse - command, non_command, poor_quality, total)
(:District - name, uuid)

Relationships:
"(State)-[:HAS_AREA]->(Area)"
"(State)-[:HAS_LOSS]->(Loss)"
"(State)-[:HAS_BLOCK_SUMMARY]->(BlockSummary)"
"(State)-[:HAS_RECHARGE]->(Recharge)"
"(State)-[:HAS_DRAFT]->(Draft)"
"(State)-[:HAS_ALLOCATION]->(Allocation)"
"(State)-[:HAS_AVAILABILITY]->(Availability)"
"(State)-[:HAS_STAGE]->(StageOfExtraction)"
"(State)-[:HAS_RAINFALL]->(Rainfall)"
"(State)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)"
"(State)-[:HAS_FUTURE_USE]->(FutureUse)"
"(State)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)"
"(State)-[:HAS_AQUIFER]->(Aquifer)"
"(State)-[:HAS_District]->(District)"

"(District)-[:HAS_AREA]->(Area)"
"(District)-[:HAS_LOSS]->(Loss)"
"(District)-[:HAS_BLOCK_SUMMARY]->(BlockSummary)"
"(District)-[:HAS_RECHARGE]->(Recharge)"
"(District)-[:HAS_DRAFT]->(Draft)"
"(District)-[:HAS_ALLOCATION]->(Allocation)"
"(District)-[:HAS_AVAILABILITY]->(Availability)"
"(District)-[:HAS_STAGE]->(StageOfExtraction)"
"(District)-[:HAS_RAINFALL]->(Rainfall)"
"(District)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)"
"(District)-[:HAS_FUTURE_USE]->(FutureUse)"
"(District)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)"
"(District)-[:HAS_AQUIFER]->(Aquifer)"

Notes:
- "India" is the only Country.
- States like Kerala, Tamil Nadu, Gujarat are (:State), not (:Country).
- Places like ernakulam, kottayam, thrissur, etc are districts of kerala.
- Convert states and districts to capital letters
- If asked for recharge_worthy or non_recharge_worthy_area, mention it in the type property of (:Area)
- If asked for total area, remember the type is "total"
- If have to use BlockSummary node, simply return the value of what is asked
"""

# --- Step 1: Convert user query to Cypher using Gemini ---
def query_to_cypher(user_query):
    prompt = f"""
    You are an assistant that converts natural language into Cypher queries.
    Schema:
    {SCHEMA}

    Convert the following user question into a Cypher query:
    "{user_query}"

    Only return the Cypher query. No explanation.
    For filtered counts, always place WHERE clause **before RETURN**.
    Remove ```cypher from the beginning and ``` from the end and return the response
    """
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Step 2: Run Cypher on Neo4j ---
def run_cypher(cypher_query):
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# --- Chatbot Interface ---
def chatbot(user_query):
    cypher = query_to_cypher(user_query)    

    try:
        results = run_cypher(cypher)
        return results if results else "No results found."
    except Exception as e:
        return f"⚠️ Error running query: {e}"
    
def generate_response(answer, query):
    prompt = f"""   
    You will be given the result of a Cypher query.
    Convert it into a meaningful sentence so that the user can understand it 
    If the given result is not a cypher query, respond non-affirmatively in a polite way

    Rules:
    Provide answers that are clear, concise, and factual.

    Style:
    Use a professional and technical tone.
    Structure responses logically for readability.
    Avoid speculation or assumptions.

    Context:
    {answer}

    Query: {query}
    Answer:
    """
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    response = model.generate_content(prompt)
    return response.text

# --- Example ---
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = chatbot(q)
        response = generate_response(answer, q)
        print("Answer:", response)