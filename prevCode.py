from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
import weaviate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import traceback

# Add token counting at the top with other imports
from transformers import GPT2Tokenizer  # <-- NEW IMPORT

# Initialize tokenizer for counting
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # <-- NEW TOKENIZER INIT

def count_tokens(text):
    """Count tokens in text using GPT-2 tokenizer"""
    return len(tokenizer.encode(text))  # <-- NEW FUNCTION

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "D:/my-workspace/models/Mistral-7B/.cache/huggingface"
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your OpenAI API key


# Custom prompt template
CUSTOM_PROMPT = PromptTemplate(
    template="""Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"],
)
def load_documents():
    try:
        loader = TextLoader("./data.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(texts)} document chunks")
        return texts
    except Exception as e:
        print(f"Error loading documents: {e}")
        traceback.print_exc()  # Added stack trace
        return []


def create_weaviate_index(docs):
    try:
        # Initialize Weaviate client for v3
        client = weaviate.Client(
            url="http://localhost:8080",
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Using OpenAI key
            }
        )

        # Test connection more thoroughly
        if not client.is_ready():
            raise ConnectionError("Weaviate connection failed")
        print("Weaviate server version:", client.get_meta())

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Added for better embeddings
        )

        # Create vector store with explicit parameters
        vector_store = Weaviate.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            index_name="LangChainIndex",
            by_text=False,
            text_key="text"  # Explicitly set text key
        )
        print("Weaviate index created successfully")
        return vector_store
    except Exception as e:
        print(f"Error creating index: {e}")
        traceback.print_exc()
        return None

def load_gpt():
    try:
        # Using OpenAI's GPT model instead of Mistral
        llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",  # or "gpt-4" if you have access
            temperature=0.3,
            max_tokens=512  # Increased for more comprehensive answers
        )
        print("GPT model loaded successfully")
        return llm
    except Exception as e:
        print(f"Error loading GPT model: {e}")
        traceback.print_exc()
        return None

def answer_question(rag_chain, query):
    try:
        print("Searching in documents...")

        # NEW: Token counting before invoking
        prompt = CUSTOM_PROMPT.format(context="[will be added]", question=query)
        token_count = count_tokens(prompt)
        print(f"Pre-check token count: {token_count}")  # <-- DEBUGGING

        result = rag_chain.invoke({"query": query})

        # NEW: Check token count in response if needed
        if hasattr(result, 'result'):
            response_tokens = count_tokens(result['result'])
            if response_tokens > 500:  # Adjust threshold as needed
                print(f"Warning: Long response ({response_tokens} tokens)")

            print("No good answer found in documents, using general knowledge...")
            general_llm = OpenAI(temperature=0.5)
            general_answer = general_llm.invoke(query)
            return f"I couldn't find a specific answer in the documents, but generally:\n\n{general_answer}"

        return f"From the documents:\n\n{result['result']}"

    except OpenAI.BadRequestError as e:  # <-- NEW SPECIFIC ERROR HANDLING
        if "maximum context length" in str(e):
            print("Token limit exceeded, suggesting more specific question")
            return "The question requires too much context. Please ask a more specific question."
        else:
            traceback.print_exc()
            return "Sorry, I encountered an error with the API."

    except Exception as e:  # <-- KEEP existing general exception handler
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error processing your question."

if __name__ == "__main__":
    try:
        print("Loading documents...")
        documents = load_documents()
        if not documents:
            print("No documents loaded - check data.txt file")
            exit(1)

        print("Creating Weaviate index...")
        vector_db = create_weaviate_index(documents)
        if not vector_db:
            print("Failed to create Weaviate index")
            exit(1)

        print("Loading GPT model...")
        llm = load_gpt()
        if not llm:
            print("Failed to load GPT model")
            exit(1)

        print("Creating RAG pipeline...")
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(
                search_kwargs={
                    "k": 2,  # Reduced from 5
                    "score_threshold": 0.7  # More selective
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": CUSTOM_PROMPT,
                "document_variable_name": "context"  # Make sure this matches your prompt
            }
        )
        print("RAG pipeline ready")

        # Test components before main loop
        print("\nTesting components:")
        test_query = "What is the main topic?"
        try:
            test_result = rag_chain.invoke({"query": test_query})
            print(f"Test query '{test_query}' returned:", test_result.get('result', 'No result'))
        except Exception as e:
            print(f"Test query failed: {e}")
            traceback.print_exc()
            exit(1)

        while True:
            try:
                query = input("\nEnter your question (or 'quit' to exit): ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break

                # NEW: Optional token check before processing
                if count_tokens(query) > 200:  # <-- PRE-FILTER LONG QUERIES
                    print("Question is too long. Please rephrase more concisely.")
                    continue

                answer = answer_question(rag_chain, query)
                print("\n" + answer)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()