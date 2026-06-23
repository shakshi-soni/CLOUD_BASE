%%writefile retrieval/query_rewriter.py
import os
from groq import Groq

def rewrite_search_query(chat_history: list, latest_query: str) -> str:
    """Combines chat history and the current prompt into a standalone search query."""
    if len(chat_history) < 2:
        return latest_query
        
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    # Format recent history for context matching
    formatted_turns = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_history[-3:]])
    
    system_prompt = (
        "You are an expert search query optimizer. Combine the chat history context and the latest user query "
        "into a single, descriptive standalone search keyword or phrase. Do NOT answer the question. "
        "Respond with ONLY the optimized search string and nothing else."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"History:\n{formatted_turns}\n\nCurrent Query: {latest_query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return latest_query
