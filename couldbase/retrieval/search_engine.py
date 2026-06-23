%%writefile retrieval/search_engine.py
import chromadb
from chromadb.utils import embedding_functions
from retrieval.query_rewriter import rewrite_search_query

class CloudDashVectorStore:
    def __init__(self):
        """Initializes the persistent client and MiniLM embedding framework."""
        self.db_client = chromadb.Client()
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.db_client.get_or_create_collection(
            name="clouddash_kb_core", 
            embedding_function=self.embed_fn
        )

    def search_knowledge_base(self, chat_history: list, user_query: str, category_filter: str = None) -> tuple:
        """Rewrites queries and extracts matching articles from the collection."""
        # 1. Optimize the incoming raw string using history context
        optimized_search_term = rewrite_search_query(chat_history, user_query)
        
        # 2. Build metadata search restrictions if supplied
        where_clause = {"category": category_filter} if category_filter else None
        
        # 3. Execute vector matching
        results = self.collection.query(
            query_texts=[optimized_search_term],
            n_results=1,
            where=where_clause
        )
        
        # 4. Extract content and unique document key identifier strings
        if results['documents'] and results['documents'][0]:
            document_text = results['documents'][0][0]
            document_id = results['metadatas'][0][0]['id']
            return document_text, document_id
            
        return "", ""
