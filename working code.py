from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
import weaviate
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.chains import RetrievalQA, LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import traceback



# Initialize tokenizer for counting
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def count_tokens(text):
    """Count tokens in text using GPT-2 tokenizer"""
    return len(tokenizer.encode(text))


# Set Hugging Face cache directory
os.environ["HF_HOME"] = "D:/my-workspace/models/Mistral-7B/.cache/huggingface"
os.environ["OPENAI_API_KEY"] = "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"

# Walden University website base URL
WALDEN_BASE_URL = "https://www.waldenu.edu"

# Custom prompt template with Walden-specific guidance
WALDEN_PROMPT = PromptTemplate(
    template="""You are a Walden University admissions assistant. Provide specific information only if relevant to the question.

Context: {context}
Question: {question}

Follow these rules:
2. For program queries, provide exact details from context
3. For vague queries, ask for clarification
4. Never hallucinate information
5. Format responses clearly with bullet points when listing items

Response:""",
    input_variables=["context", "question"],
)

GENERAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant for Walden University. Answer the following question.
If it's related to Walden University programs but you're not sure, say you'll check the website.
If it's a general greeting, respond politely.
If it's completely unrelated, say you can only help with Walden University information.

Question: {question}
Answer:"""
)

def load_documents():
    try:
        loader = TextLoader("./data.txt", encoding='utf-8')
        documents = loader.load()

        # Improved text splitting for better chunking
        text_splitter = CharacterTextSplitter(
            chunk_size=300,  # Increased for better context
            chunk_overlap=50,
            separator="\n"
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def create_weaviate_index(docs):
    try:
        client = weaviate.Client(url="http://localhost:8080")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        return Weaviate.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            index_name="WaldenPrograms"
        )
        print("Weaviate index created successfully")
        return vector_store
    except Exception as e:
        print(f"Error creating index: {e}")
        traceback.print_exc()
        return None


def load_gpt():
    return OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.3,
        max_tokens=300
    )

def scrape_walden_website(query):
    """Scrape Walden University website for relevant information"""
    try:
        print("Scraping query:", query)
        print("WALDEN_BASE_URL", WALDEN_BASE_URL)

        # Construct the search URL
        search_url = f"{WALDEN_BASE_URL}/search?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # print(f"Searching Walden website for: {query}")
        response = requests.get(search_url, headers=headers, timeout=10)

        # Check if request was successful
        response.raise_for_status()

        # print("Walden website response", response)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the most relevant program page
        # program_link = None
        # for link in soup.select('a[href*="/programs/"]'):
        #     if 'program' in link.get_text(strip=True).lower():
        #         program_link = urljoin(WALDEN_BASE_URL, link['href'])
        #         break

        program_link = None
        for link in soup.select('a[href]'):  # Select all links
            page_url = urljoin(WALDEN_BASE_URL, link['href'])

            # Ensure it's an internal Walden University link
            if page_url.startswith(WALDEN_BASE_URL):
                program_link = page_url  # Store the first relevant link found
                break  # Stop at the first match (or remove `break` to collect multiple links)

        # print(f"Program link: {program_link}")

        if not program_link:
            print("No relevant program link found.")
            return None

        # Scrape the program page
        program_response = requests.get(program_link, headers=headers, timeout=10)
        program_response.raise_for_status()
        # print("Program response", program_response)
        program_soup = BeautifulSoup(program_response.text, 'html.parser')
        # print("Program soup", program_soup)

        # Extract key information
        title_element = program_soup.find('h1')
        # print("Title element", title_element)
        title = title_element.get_text(strip=True) if title_element else "Program Information"
        # print(f"Title: {title}")
        content_div = program_soup.find('div', class_='program-details') or program_soup.find('main')
        # print("Content div", content_div)
        if content_div:
            paragraphs = content_div.find_all('p')[:5]
            # print("Paragraphs:", paragraphs)
            content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            # print("Content:", content)
        else:
            content = "No details found"

        return {
            'title': title,
            'content': content,
            'url': program_link
        }

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None
    except Exception as e:
        print(f"Error scraping website: {e}")
        return None


def answer_question(rag_chain, llm, query):
    query = query.strip()

    # Handle greetings
    if query.lower() in ['hi', 'hello', 'hey']:
        return "Hello! I'm Walden University's assistant. How can I help you with program information today?"

    try:
        # First try to find answer in documents
        print("Searching in documents...")
        result = rag_chain.invoke({"query": query})

        # print("result::::::::::::",result)

        # Check if we got a meaningful answer
        if (not result['result'] or
            "i don't know" in result['result'].lower() or
            len(result['result'].strip()) < 3000):

            # print("query::::::::::",query.lower())

            # Check if it's Walden-related
            if any(word in query.lower() for word in ['walden', 'program', 'admission', 'tuition', 'course', 'degree']):
                print("No answer found, scraping Walden website.....")
                scraped_data = scrape_walden_website(query)

                # print("scraped_data",scraped_data)

                if scraped_data:
                    prompt = ChatPromptTemplate.from_template(
                        """Give this Walden University program information in a helpful way respond in bullet points:

                        Title: {title}
                        Content: {content}

                        Create a concise response that answers: {query}
                        Include this URL for more details: {url}
                        Response:"""
                    )

                    chain = (
                            prompt
                            | llm  # Pipe the prompt directly to the LLM
                    )
                    response = chain.invoke({
                        "title": scraped_data['title'],
                        "content": scraped_data['content'],
                        "query": query,
                        "url": scraped_data['url']
                    })

                    return response
                else:
                    return "I couldn't find specific information on Walden's website. Please visit waldenu.edu for more details."
            else:
                # Use GPT for general knowledge
                print("Use GPT for general information...................")

                GENERAL_PROMPT = ChatPromptTemplate.from_template(
                    """You are an academic advisor for Walden University, providing information on programs, admissions, and career development  respond in bullet points.

                    If the question relates to **education, admissions, career development, or academic programs**, your response should:
                    - **Primarily focus on what Walden University offers** in this context.
                    - Provide details on relevant programs, degrees, and resources at Walden.
                    - If the exact program is not available, suggest similar options Walden provides.
                    - Share guidance on admissions, career paths, or professional growth within Walden’s academic framework.

                    **Walden University Overview:**
                    - Known for its **online education** and **mission of social change**.
                    - Offers **undergraduate, graduate, and doctoral programs** in **business, education, psychology, nursing, public policy**, and more.
                    - **Flexible learning options** for working professionals and international students.
                    - **Emphasizes applied research, real-world learning, and career advancement.**

                    **If the question is NOT related to education, admissions, or career, provide a general informative response.**  

                    Question: {question}

                    Answer:
                    """
                )

                general_chain = GENERAL_PROMPT | llm
                return general_chain.invoke({"question": query})

        # print("result 2 ??????????",result['result'])

        return result['result']

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I encountered an error. Please try again."

if __name__ == "__main__":
    try:
        print("Initializing system...")
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
            retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": WALDEN_PROMPT}
        )

        print("Walden University Assistant ready!")
        while True:
            query = input("\nAsk about programs (or 'quit'): ").strip()
            if query.lower() == 'quit':
                break

            if not query:
                continue

            answer = answer_question(rag_chain, llm, query)
            print("\n" + answer)

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()