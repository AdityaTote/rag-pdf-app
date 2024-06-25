from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    return embeddings