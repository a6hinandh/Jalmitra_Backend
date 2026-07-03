# delete_index.py

from pinecone import Pinecone

# Replace with your actual API key
API_KEY = ""

pc = Pinecone(api_key=API_KEY)

# Delete the incorrectly created index
try:
    pc.delete_index('my-index')
    print("✅ Successfully deleted ''")
except Exception as e:
    print(f"Error deleting index: {e}")

# List remaining indexes
indexes = pc.list_indexes().names()
print("Remaining indexes:", indexes)