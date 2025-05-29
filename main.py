from together import Together
from config import TOGETHER_API_KEY, MODEL_NAME, SEMANTIC_WEIGHT, SEARCH_TYPE
from prompt_engineering import SYSTEM_PROMPT
from vector_store import HybridMarineEdgeVectorStore

# Initialize Together client
client = Together(api_key=TOGETHER_API_KEY)

# Initialize hybrid vector store
vector_store = HybridMarineEdgeVectorStore(pdf_directory="./pdfs")
vector_store.create_or_load_vector_store()


def get_relevant_context(query, k=5, search_type=SEARCH_TYPE):
    """Retrieve relevant information using hybrid search"""
    results = vector_store.query_vector_store(query, k=k, search_type=search_type)

    # Format the retrieved content
    context_texts = []
    for doc, score in results:
        if score > 0.1:  # Lower threshold for hybrid search
            context_texts.append(f"--- Relevant Information (score: {score:.2f}) ---\n{doc.page_content}")

    return "\n\n".join(context_texts)


def get_bot_response(user_input):
    """Get response from the bot using Together AI with hybrid RAG"""
    try:
        # Get relevant context from hybrid vector store
        relevant_context = get_relevant_context(user_input)

        # Create enhanced prompt with context
        if relevant_context:
            enhanced_input = f"""
RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{relevant_context}

USER QUERY: {user_input}

Please provide a helpful response based on the above context. If the context contains specific numbers, lists, or detailed information that directly answers the question, make sure to include ALL relevant details.
"""
        else:
            enhanced_input = user_input

        # Generate response using Together AI
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_input}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error: {str(e)}")
        return "I'm sorry, I encountered an error processing your request. Please try again or rephrase your question."


def main():
    """Command-line interface for testing hybrid RAG with Together AI"""
    print("Marine Edge Assistant with Hybrid RAG + Together AI (Type 'exit' to quit)")
    print("-" * 70)
    print("Initializing hybrid vector database... Please wait.")

    print("Together AI bot initialized successfully!")
    print(f"Search mode: {SEARCH_TYPE}")
    print(f"Semantic weight: {SEMANTIC_WEIGHT}")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using Marine Edge Assistant!")
            break

        # Test different search modes
        if user_input.startswith("/semantic"):
            query = user_input.replace("/semantic", "").strip()
            print(f"\n[SEMANTIC SEARCH] Results for: {query}")
            context = get_relevant_context(query, search_type="semantic")
            print(context)
            continue
        elif user_input.startswith("/keyword"):
            query = user_input.replace("/keyword", "").strip()
            print(f"\n[KEYWORD SEARCH] Results for: {query}")
            context = get_relevant_context(query, search_type="keyword")
            print(context)
            continue

        bot_response = get_bot_response(user_input)
        print(f"\nMarine Edge Assistant: {bot_response}")


if __name__ == "__main__":
    main()