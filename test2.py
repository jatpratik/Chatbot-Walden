from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
import weaviate
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import traceback

# Initialize tokenizer for token counting
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def count_tokens(text):
    return len(tokenizer.encode(text))

# Set environment variables
os.environ["HF_HOME"] = "D:/my-workspace/models/Mistral-7B/.cache/huggingface"
os.environ["OPENAI_API_KEY"] = "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"  # Replace with actual key

# Custom Prompt Template
CUSTOM_PROMPT = PromptTemplate(
    template="""Context: {context}\nQuestion: {question}\nAnswer:""",
    input_variables=["context", "question"],
)

def load_documents():
    try:
        loader = TextLoader("./data.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(texts)} document chunks")
        return texts
    except Exception as e:
        print(f"Error loading documents: {e}")
        traceback.print_exc()
        return []

def create_weaviate_index(docs):
    try:
        client = weaviate.Client(url="http://localhost:8080")
        if not client.is_ready():
            raise ConnectionError("Weaviate connection failed")
        print("Weaviate connected successfully")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Weaviate.from_documents(
            documents=docs, embedding=embeddings, client=client, index_name="LangChainIndex"
        )
        return vector_store
    except Exception as e:
        print(f"Error creating index: {e}")
        traceback.print_exc()
        return None

def load_gpt():
    try:
        return OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.3, max_tokens=512)
    except Exception as e:
        print(f"Error loading GPT model: {e}")
        traceback.print_exc()
        return None

def fetch_walden_info(query):
    try:
        search_url = f"https://www.waldenu.edu/search?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("p")
            return "\n".join([res.text for res in results[:3]])
        return "No additional information found on Walden University's website."
    except Exception as e:
        print(f"Web scraping error: {e}")
        return "Could not retrieve data from Walden University website."

def answer_question(rag_chain, query):
    try:
        result = rag_chain.invoke({"query": query})
        if not result['result'] or len(result['result']) < 10:
            print("No good answer found in documents, searching online...")
            return fetch_walden_info(query)
        return result['result']
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return "Error processing your question."

if __name__ == "__main__":
    documents = load_documents()
    if not documents:
        print("No documents loaded - check data.txt file")
        exit(1)

    vector_db = create_weaviate_index(documents)
    if not vector_db:
        print("Failed to create Weaviate index")
        exit(1)

    llm = load_gpt()
    if not llm:
        print("Failed to load GPT model")
        exit(1)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.7}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT, "document_variable_name": "context"}
    )

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        if count_tokens(query) > 200:
            print("Question is too long. Please rephrase.")
            continue
        answer = answer_question(rag_chain, query)
        print("\n" + answer)
