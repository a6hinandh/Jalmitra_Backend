from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone(api_key):
    # Initialize the Pinecone client with your API key
    pc = Pinecone(api_key=api_key)

    # List existing indexes
    indexes = pc.list_indexes().names()
    print("Existing indexes:", indexes)

    # Define a valid index name (lowercase, alphanumeric, hyphens only)
    index_name = 'my-index'  # ✅ valid name

    # Create the index if it doesn't exist
    if index_name not in indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # Replace with your actual dimension size
            metric='cosine',  # or 'euclidean'
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # ✅ correct AWS region format
            )
        )
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

    # Return the index object for further operations
    index = pc.Index(index_name)
    return index
