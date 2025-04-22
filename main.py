from transformers import AutoTokenizer
import torch
import os
import weaviate
import tiktoken
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "D:/my-workspace/models/Mistral-7B/.cache/huggingface"
os.environ[
    "OPENAI_API_KEY"] = "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"

# Initialize tokenizer for context length management
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using OpenAI's tokenizer"""
    return len(tokenizer.encode(text))


def trim_context(context: str, max_tokens: int = 3500) -> str:
    """Trim context to fit within token limits"""
    tokens = tokenizer.encode(context)
    if len(tokens) <= max_tokens:
        return context
    logger.warning(f"Context too long ({len(tokens)} tokens), trimming to {max_tokens}")
    return tokenizer.decode(tokens[:max_tokens])


# Optimized prompt template
CUSTOM_PROMPT = PromptTemplate(
    template="""Answer based on context. If unsure, say "I couldn't find a specific answer..." and provide general knowledge.

Context: {context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)


def load_documents():
    """Load and split documents with smaller chunks"""
    try:
        loader = TextLoader("./data.txt")
        documents = loader.load()
        # Smaller chunks with less overlap
        text_splitter = CharacterTextSplitter(
            chunk_size=150,  # Reduced from 200
            chunk_overlap=20,  # Reduced from 30
            length_function=count_tokens  # Use token count for splitting
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Loaded {len(texts)} document chunks")
        return texts
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        traceback.print_exc()
        return []


def create_weaviate_index(docs):
    """Create Weaviate vector store with improved error handling"""
    try:
        client = weaviate.Client(
            url="http://localhost:8080",
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )

        if not client.is_ready():
            raise ConnectionError("Weaviate connection failed")
        logger.info(f"Weaviate server version: {client.get_meta()}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_store = Weaviate.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            index_name="LangChainIndex",
            by_text=False,
            text_key="text"
        )
        logger.info("Weaviate index created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        traceback.print_exc()
        return None


def load_gpt():
    """Load GPT model with context length awareness"""
    try:
        llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.3,
            max_tokens=512,
            request_timeout=30  # Added timeout
        )
        logger.info("GPT model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error loading GPT model: {e}")
        traceback.print_exc()
        return None


def create_rag_chain(llm, vector_db):
    """Create RAG chain with controlled context size"""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_kwargs={
                "k": 2,  # Only retrieve 2 most relevant documents
                "score_threshold": 0.7  # Higher similarity threshold
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": CUSTOM_PROMPT,
            "document_variable_name": "context"
        }
    )


def answer_question(rag_chain, query):
    """Answer question with context length management"""
    try:
        logger.info(f"Processing query: {query}")

        # First retrieve documents separately to manage context size
        retriever = rag_chain.retriever
        relevant_docs = retriever.get_relevant_documents(query)

        # Combine and trim context if needed
        combined_context = "\n".join([doc.page_content for doc in relevant_docs])
        token_count = count_tokens(combined_context)
        logger.info(f"Initial context tokens: {token_count}")

        if token_count > 3500:
            combined_context = trim_context(combined_context)
            logger.info(f"Trimmed context tokens: {count_tokens(combined_context)}")

        # Invoke with controlled context
        result = rag_chain.invoke({
            "query": query,
            "context": combined_context
        })

        # Handle uncertain answers
        if (result['result'].strip().lower().startswith(("i don't know", "i couldn't find")) or
                len(result['result'].strip()) < 20):

            logger.info("No confident answer found, using general knowledge")
            general_llm = OpenAI(temperature=0.5, max_tokens=256)
            general_answer = general_llm.invoke(query)
            return f"I couldn't find a specific answer in the documents, but generally:\n\n{general_answer}"

        return f"From the documents:\n\n{result['result']}"

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error processing your question."


def main():
    try:
        logger.info("Starting RAG pipeline setup...")

        # Document loading
        logger.info("Loading documents...")
        documents = load_documents()
        if not documents:
            logger.error("No documents loaded - check data.txt file")
            exit(1)

        # Vector store creation
        logger.info("Creating Weaviate index...")
        vector_db = create_weaviate_index(documents)
        if not vector_db:
            logger.error("Failed to create Weaviate index")
            exit(1)

        # Model loading
        logger.info("Loading GPT model...")
        llm = load_gpt()
        if not llm:
            logger.error("Failed to load GPT model")
            exit(1)

        # RAG pipeline
        logger.info("Creating RAG pipeline...")
        rag_chain = create_rag_chain(llm, vector_db)
        logger.info("RAG pipeline ready")

        # Test query
        logger.info("\nTesting components...")
        test_query = "What is the main topic?"
        try:
            test_result = rag_chain.invoke({"query": test_query})
            logger.info(f"Test query '{test_query}' returned: {test_result.get('result', 'No result')}")
        except Exception as e:
            logger.error(f"Test query failed: {e}")
            traceback.print_exc()
            exit(1)

        # Main interaction loop
        while True:
            try:
                query = input("\nEnter your question (or 'quit' to exit): ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break

                answer = answer_question(rag_chain, query)
                print("\n" + answer)

            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
            except Exception as e:
                logger.error(f"\nError processing query: {e}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()