# query_index.py

from sentence_transformers import SentenceTransformer
from pinecone_setup import initialize_pinecone

def query_index(index, model, query_text, top_k=5):
    query_vector = model.encode([query_text])[0].tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"
    index = initialize_pinecone(API_KEY)
    model = SentenceTransformer('all-mpnet-base-v2')

    query_text = input("Enter your query: ")
    results = query_index(index, model, query_text)

    print("✅ Query Results:")
    for match in results['matches']:
        print("ID:", match['id'])
        print("Details:", match['metadata']['text'])
        print("---")
