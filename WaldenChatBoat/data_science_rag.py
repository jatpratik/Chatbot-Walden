import os
import json
import logging
from typing import List, Dict, Any
import weaviate
from sentence_transformers import SentenceTransformer
import datetime

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage
# from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='walden_rag.log')
logger = logging.getLogger('walden_rag')


class WaldenRAGChain:
    def __init__(self,
                 weaviate_url="http://localhost:8080",
                 api_key=None,
                 embedding_model="all-MiniLM-L6-v2",
                 openai_api_key=None,
                 model_name="gpt-3.5-turbo"):

        self.class_name = "WaldenContent"
        self.weaviate_url = weaviate_url
        self.api_key = api_key

        # Initialize memory for chat history
        self.memory = {
            "chat_history": ChatMessageHistory()
        }

        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            # Check if OPENAI_API_KEY is already set
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OpenAI API key must be provided or set as environment variable")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.model = SentenceTransformer(embedding_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Connect to Weaviate
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        try:
            if api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=weaviate_url)

            # Check if Weaviate is ready
            if not self.client.is_ready():
                logger.error("Weaviate is not ready")
                raise ConnectionError("Weaviate server is not ready")
            logger.info("Successfully connected to Weaviate")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)

        # Initialize QA chain
        self.initialize_qa_chain()

        # Initialize answer refiner chain
        self.initialize_answer_refiner()

        # Initialize query rewriter chain
        self.initialize_query_rewriter()

        # Initialize fallback chain for when no context is found
        self.initialize_fallback_chain()

    def initialize_query_rewriter(self):
        """Initialize the query rewriter chain to create standalone queries"""
        query_rewriter_template = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. 

        DO NOT answer the question, just reformulate it if needed and otherwise return it as is.

        CHAT HISTORY:
        {chat_history}

        LATEST QUESTION: {question}

        STANDALONE QUESTION:"""

        QUERY_REWRITER_PROMPT = PromptTemplate(
            template=query_rewriter_template,
            input_variables=["question", "chat_history"]
        )

        self.query_rewriter_chain = (
                QUERY_REWRITER_PROMPT
                | self.llm
                | StrOutputParser()
        )

    def initialize_qa_chain(self):
        """Initialize the QA chain with a comprehensive prompt"""
        qa_template = """You are a helpful AI assistant for Walden University, specifically designed to provide information about the university's programs, 
        courses, admissions, and other related topics. Use the following pieces of context to answer the question at the end. Be helpful, accurate, and friendly.

        CONTEXT:
        {context}

        CHAT HISTORY:
        {chat_history}

        CURRENT QUESTION: {question}

        INSTRUCTIONS:
        1. If the current question refers to previous questions or answers, use the chat history to understand the context.
        2. Focus on answering the specific question using the provided context.
        3. If the question asks about something not covered in the context but was discussed in the chat history, use that information to answer.
        4. If you don't have enough information to provide a complete answer, be honest about what you do and don't know.

        ANSWER:
        """

        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question", "chat_history"]
        )

        self.qa_chain = (
                {
                    "context": lambda x: "\n\n".join(x["context"]),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: self.get_recent_chat_history(x.get("chat_history", []))
                }
                | QA_PROMPT
                | self.llm
                | StrOutputParser()
        )

    def initialize_fallback_chain(self):
        """Initialize a fallback chain for when no context is found but we still want to process using chat history"""
        fallback_template = """You are a helpful AI assistant for Walden University. The user has asked a question that doesn't match directly with our knowledge base,
        but you may be able to help based on previous conversation or general knowledge about universities.

        CHAT HISTORY:
        {chat_history}

        CURRENT QUESTION: {question}

        INSTRUCTIONS:
        1. If the current question relates to something mentioned in chat history, use that information to provide a helpful answer.
        2. For general greetings or simple questions, respond naturally.
        3. If you truly cannot provide any helpful information, be honest about your limitations.

        ANSWER:
        """

        FALLBACK_PROMPT = PromptTemplate(
            template=fallback_template,
            input_variables=["question", "chat_history"]
        )

        self.fallback_chain = (
                {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: self.get_recent_chat_history(x.get("chat_history", []))
                }
                | FALLBACK_PROMPT
                | self.llm
                | StrOutputParser()
        )

    def format_chat_history(self, messages):
        """Format chat history for inclusion in prompts"""
        if not messages:
            return "No previous conversation."

        formatted_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")

        return "\n".join(formatted_history)

    def initialize_answer_refiner(self):
        """Initialize the answer refiner chain"""
        refiner_template = """You are an AI assistant for Walden University. 
        You provide comprehensive, accurate, and helpful information about Walden University's programs, admissions, faculty, 
        student life, and other university-related topics.

        Refine the INITIAL ANSWER to ensure the FINAL ANSWER is:

        - Structured with clear **headings** and **bullet points**
        - Easy to scan and read quickly
        - Factually accurate and helpful
        - Professional and friendly in tone

        ---
        **QUESTION:** {question}

        **INITIAL ANSWER:** {answer}

        ---
        **FINAL ANSWER:** 
        - Use markdown-style formatting (e.g., `###`, `-`, `1.`)
        - Add a short closing line inviting follow-up questions
        """

        REFINER_PROMPT = PromptTemplate(
            template=refiner_template,
            input_variables=["question", "answer"]
        )

        self.refiner_chain = (
                REFINER_PROMPT
                | self.llm
                | StrOutputParser()
        )

    def generate_embedding(self, text):
        """Generate vector embedding for text using SentenceTransformer"""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def retrieve_context(self, query, limit=5):
        """Retrieve relevant context from Weaviate"""
        try:
            # Generate query vector
            query_vector = self.generate_embedding(query)

            if not query_vector:
                logger.error("Failed to generate query embedding")
                return []

            # Execute the search
            result = self.client.query.get(
                class_name=self.class_name,
                properties=["doc_id", "url", "text", "source", "url_path"]
            ).with_near_vector({
                "vector": query_vector,
                "certainty": 0.65  # Reduced the certainty threshold for better recall
            }).with_limit(limit).do()

            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                return result["data"]["Get"][self.class_name]
            else:
                logger.warning(f"Unexpected query result structure: {result}")
                return []
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    def get_recent_chat_history(self, messages, max_messages=4):
        """Return only the most recent messages from chat history, limited to max_messages"""
        if not messages or len(messages) == 0:
            return "No previous conversation."

        # Get only the most recent messages (up to max_messages)
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

        formatted_history = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")

        return "\n".join(formatted_history)

    def process_query(self, query, user_id="user", refine_answer=True):
        """Process a user query through the RAG pipeline"""
        try:
            # Log the query
            logger.info(f"Processing query: '{query}' for user {user_id}")

            # Handle simple greetings separately
            if query.lower().strip() in ["hi", "hello", "hey"]:
                greeting_response = "Hello! How can I help you with information about Walden University today?"
                self.memory["chat_history"].add_user_message(query)
                self.memory["chat_history"].add_ai_message(greeting_response)

                return {
                    "answer": greeting_response,
                    "sources": [],
                    "metadata": {
                        "context_chunks": 0,
                        "processing_time": 0,
                        "has_relevant_info": True,
                        "is_greeting": True
                    }
                }

            # Rewrite the query to make it standalone if there's chat history
            start_time = datetime.datetime.now()
            original_query = query

            if len(self.memory["chat_history"].messages) > 0:
                # Only rewrite if we have chat history

                # Get only the last message pair (question and answer)
                latest_messages = self.memory["chat_history"].messages[-2:] if len(
                    self.memory["chat_history"].messages) >= 2 else self.memory["chat_history"].messages

                logger.info(f" hello history ::::::::::::::::::::::::: '{self.memory["chat_history"].messages}' for")
                rewritten_query = self.query_rewriter_chain.invoke({
                    "question": query,
                    "chat_history": self.format_chat_history(latest_messages)
                })
                query = rewritten_query
                logger.info(f"Rewrote query from '{original_query}' to '{query}'")

            rewrite_time = (datetime.datetime.now() - start_time).total_seconds()

            # Retrieve relevant context
            start_time = datetime.datetime.now()
            context_chunks = self.retrieve_context(query)
            logger.info(f"Data from vectore DB: '{context_chunks}' >>>>>>>>>>>>>>")
            retrieval_time = (datetime.datetime.now() - start_time).total_seconds()

            # Extract text from context chunks
            context_texts = [chunk["text"] for chunk in context_chunks]

            # Track sources for citation
            sources = [{"url": chunk["url"], "doc_id": chunk["doc_id"]} for chunk in context_chunks]

            # Log context retrieval
            logger.info(f"Retrieved {len(context_chunks)} context chunks in {retrieval_time:.2f} seconds")

            if not context_chunks:
                # No relevant context found, use fallback chain that relies on chat history
                start_time = datetime.datetime.now()
                with get_openai_callback() as cb:
                    fallback_response = self.fallback_chain.invoke({
                        "question": original_query,  # Use original query for fallback
                        "chat_history": self.memory["chat_history"].messages
                    })

                fallback_time = (datetime.datetime.now() - start_time).total_seconds()
                logger.info(f"Generated fallback answer in {fallback_time:.2f} seconds, tokens: {cb.total_tokens}")

                # Update memory
                self.memory["chat_history"].add_user_message(original_query)
                self.memory["chat_history"].add_ai_message(fallback_response)

                return {
                    "answer": fallback_response,
                    "sources": [],
                    "metadata": {
                        "context_chunks": 0,
                        "processing_time": fallback_time + rewrite_time,
                        "has_relevant_info": False,
                        "used_fallback": True,
                        "original_query": original_query,
                        "rewritten_query": query if query != original_query else None
                    }
                }

            # Generate answer using QA chain
            start_time = datetime.datetime.now()
            with get_openai_callback() as cb:
                qa_response = self.qa_chain.invoke({
                    "context": context_texts,
                    "question": original_query,  # Use original query for QA
                    "chat_history": self.memory["chat_history"].messages
                })

            qa_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Generated initial answer in {qa_time:.2f} seconds, tokens: {cb.total_tokens}")

            # Refine answer if requested
            if refine_answer:
                start_time = datetime.datetime.now()
                refiner_response = self.refiner_chain.invoke({"question": original_query, "answer": qa_response})
                refine_time = (datetime.datetime.now() - start_time).total_seconds()
                final_answer = refiner_response
                logger.info(f"Refined answer in {refine_time:.2f} seconds")
            else:
                final_answer = qa_response
                refine_time = 0

            # Update memory
            self.memory["chat_history"].add_user_message(original_query)
            self.memory["chat_history"].add_ai_message(final_answer)

            # Construct the response
            response = {
                "answer": final_answer,
                "sources": sources,
                "metadata": {
                    "context_chunks": len(context_chunks),
                    "processing_time": rewrite_time + qa_time + refine_time,
                    "has_relevant_info": True,
                    "original_query": original_query,
                    "rewritten_query": query if query != original_query else None
                }
            }

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again or ask a different question about Walden University.",
                "sources": [],
                "metadata": {
                    "error": str(e)
                }
            }

    def handle_chat(self, message, user_id="user", conversation_id=None):
        """Handle a chat message and return a response"""
        # Process the query
        response = self.process_query(message, user_id)

        # Format the response with citations if available
        if response["sources"]:
            formatted_sources = []
            for idx, source in enumerate(response["sources"][:3], 1):  # Limit to top 3 sources
                formatted_sources.append(f"[{idx}] {source['url']}")

            sources_text = "\n\nSources:\n" + "\n".join(formatted_sources)
            response["formatted_answer"] = response["answer"] + sources_text
        else:
            response["formatted_answer"] = response["answer"]

        return response


def main():
    # For testing purposes
    import dotenv
    dotenv.load_dotenv()  # Load API keys from .env file

    # Initialize the RAG chain
    openai_api_key = os.getenv("OPENAI_API_KEY")

    rag = WaldenRAGChain(
        weaviate_url="http://localhost:8080",
        openai_api_key=openai_api_key
    )

    # Test query
    test_query = "What doctoral programs does Walden University offer?"
    print(f"\nQuestion: {test_query}")
    response = rag.handle_chat(test_query)
    print("\nAnswer:")
    print(response["formatted_answer"])

    # Test follow-up question
    follow_up = "What are the admission requirements for these programs?"
    print(f"\nFollow-up Question: {follow_up}")
    response = rag.handle_chat(follow_up)
    print("\nAnswer:")
    print(response["formatted_answer"])


if __name__ == "__main__":
    main()