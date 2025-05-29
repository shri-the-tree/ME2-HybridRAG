import logging
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from together import Together
from config import TOGETHER_API_KEY, MODEL_NAME, TELEGRAM_TOKEN, PDF_DIRECTORY, VECTOR_DB_DIRECTORY
from prompt_engineering import SYSTEM_PROMPT
from vector_store import HybridMarineEdgeVectorStore

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("telegram_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Together AI
client = Together(api_key=TOGETHER_API_KEY)

# Initialize hybrid vector store
logger.info("Initializing hybrid vector store...")
vector_store = HybridMarineEdgeVectorStore(
    pdf_directory=PDF_DIRECTORY,
    persist_directory=VECTOR_DB_DIRECTORY
)
vector_store.create_or_load_vector_store()
logger.info(f"Hybrid vector store loaded with {len(vector_store.documents)} documents")

# Dictionary to store conversation history for each user
user_conversations = {}


def clean_response(text):
    """Remove thinking tags and clean up response"""
    # Remove <think> blocks entirely
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove any remaining thinking indicators
    text = re.sub(r'^\s*<think>.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*</think>.*?$', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()

    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Hi! I'm the Marine Edge Assistant. I can help you with questions about IMUCET, "
        "DNS sponsorship, and Marine Edge courses. How can I assist you today?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "You can ask me about:\n"
        "- IMUCET exam details and dates\n"
        "- Eligibility criteria for marine courses\n"
        "- Marine Edge course offerings\n"
        "- Preparation strategies\n"
        "- DNS sponsorship information\n\n"
        "Just type your question and I'll do my best to help!"
    )


def get_relevant_context(query, k=5):
    """Retrieve relevant information using hybrid search"""
    try:
        results = vector_store.query_vector_store(query, k=k, search_type="hybrid")

        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            return None

        # Format the retrieved content
        context_texts = []
        for doc, score in results:
            if score > 0.1:  # Relevance threshold
                source = doc.metadata.get('source', 'Marine Edge Knowledge Base')
                content = doc.page_content.strip()

                if len(content) > 50:  # Minimum content length
                    context_texts.append(f"SOURCE: {source}\nCONTENT: {content}")

        if not context_texts:
            return None

        logger.info(f"Retrieved {len(context_texts)} relevant document chunks")
        return "\n\n".join(context_texts)

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return None


async def get_bot_response(user_id: int, user_input: str) -> str:
    """Get response from the bot using Together AI with hybrid RAG"""
    try:
        # Get or initialize conversation history for this user
        if user_id not in user_conversations:
            user_conversations[user_id] = []

        logger.info(f"User {user_id} query: {user_input}")

        # Retrieve relevant context using hybrid search
        relevant_context = get_relevant_context(user_input, k=5)

        # Create enhanced prompt with context
        if relevant_context:
            enhanced_input = f"""
RELEVANT CONTEXT FROM MARINE EDGE KNOWLEDGE BASE:
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
            ],
            temperature=0.2,
            max_tokens=1024
        )

        bot_response = response.choices[0].message.content

        # Clean up thinking tags
        bot_response = clean_response(bot_response)

        # Add to conversation history
        user_conversations[user_id].append({"role": "user", "content": user_input})
        user_conversations[user_id].append({"role": "assistant", "content": bot_response})

        # Keep history within reasonable limits
        if len(user_conversations[user_id]) > 10:
            user_conversations[user_id] = user_conversations[user_id][-10:]

        logger.info(f"Generated response for user {user_id}: {len(bot_response)} characters")
        return bot_response

    except Exception as e:
        error_msg = f"Error generating response for user {user_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return "I'm sorry, I encountered an error processing your request. Please try again or rephrase your question."


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_id = update.effective_user.id
    user_input = update.message.text
    username = update.effective_user.username or "Unknown"

    logger.info(f"Received message from user {user_id} ({username}): {user_input}")

    # Show typing indicator while generating response
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    # Get response from the bot
    response = await get_bot_response(user_id, user_input)

    # Send the response
    await update.message.reply_text(response)
    logger.info(f"Sent response to user {user_id} ({username})")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show system statistics to admin users."""
    admin_ids = [12345678]  # Replace with actual admin user IDs
    user_id = update.effective_user.id

    if user_id in admin_ids:
        stats = f"System Statistics:\n"
        stats += f"- Active users: {len(user_conversations)}\n"
        stats += f"- Documents in hybrid vector store: {len(vector_store.documents)}\n"
        stats += f"- FAISS index size: {vector_store.faiss_index.ntotal if vector_store.faiss_index else 0}\n"

        await update.message.reply_text(stats)
    else:
        await update.message.reply_text("Sorry, this command is only available to administrators.")


def main() -> None:
    """Start the bot."""
    try:
        # Create the Application
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Start the Bot
        logger.info("Starting Marine Edge Telegram bot with Hybrid RAG...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Failed to start bot: {str(e)}")


if __name__ == "__main__":
    main()