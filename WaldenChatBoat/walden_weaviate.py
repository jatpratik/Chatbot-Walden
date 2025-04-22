import weaviate
import json
import os
from weaviate.auth import AuthApiKey
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='walden_weaviate.log')
logger = logging.getLogger('walden_weaviate')


class WaldenWeaviateDB:
    def __init__(self, data_folder="walden_data",
                 weaviate_url="http://localhost:8080",
                 api_key=None,
                 embedding_model="all-MiniLM-L6-v2"):
        self.data_folder = data_folder
        self.chunks_file = os.path.join(data_folder, "rag_chunks.jsonl")
        self.class_name = "WaldenContent"

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
                auth_config = AuthApiKey(api_key=api_key)
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

    def create_schema(self):
        """Create the schema for Walden University content"""
        # Check if class already exists
        try:
            schema = self.client.schema.get()
            classes = [cls['class'] for cls in schema['classes']] if 'classes' in schema else []

            if self.class_name in classes:
                logger.info(f"Class {self.class_name} already exists. Deleting it to recreate.")
                self.client.schema.delete_class(self.class_name)
        except Exception as e:
            logger.warning(f"Error checking schema: {e}")

        # Define class schema
        class_obj = {
            "class": self.class_name,
            "description": "Walden University content for the RAG chatbot",
            "vectorizer": "none",  # We'll add our own vectors
            "properties": [
                {
                    "name": "doc_id",
                    "dataType": ["string"],
                    "description": "Unique document ID"
                },
                {
                    "name": "url",
                    "dataType": ["string"],
                    "description": "Source URL of the content"
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The content text chunk"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Source identifier"
                },
                {
                    "name": "url_path",
                    "dataType": ["string"],
                    "description": "URL path component",
                    "indexSearchable": True
                }
            ]
        }

        # Create the class
        try:
            logger.info(f"Creating schema class: {self.class_name}")
            self.client.schema.create_class(class_obj)
            logger.info(f"Schema class {self.class_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False

    def generate_embedding(self, text):
        """Generate vector embedding for text using SentenceTransformer"""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def load_data(self, batch_size=50):
        """Load the chunked data into Weaviate"""
        try:
            # Check if chunks file exists
            if not os.path.exists(self.chunks_file):
                logger.error(f"Chunks file not found: {self.chunks_file}")
                raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")

            # Read chunks
            chunks = []
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))

            logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file}")

            # Process in batches
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            success_count = 0

            with tqdm(total=len(chunks), desc="Importing chunks to Weaviate") as pbar:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]

                    # Process each document in the batch
                    with self.client.batch as batch_processor:
                        # Configure batch
                        batch_processor.batch_size = batch_size
                        batch_processor.callback = None  # Optional progress callback

                        for item in batch:
                            # Generate embedding
                            text = item["text"]
                            vector = self.generate_embedding(text)

                            if vector:
                                try:
                                    # Create object with vector
                                    properties = {
                                        "doc_id": item["doc_id"],
                                        "url": item["url"],
                                        "text": text,
                                        "source": item["source"],
                                        "url_path": item["url_path"]
                                    }

                                    batch_processor.add_data_object(
                                        properties,
                                        self.class_name,
                                        vector=vector
                                    )
                                    success_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to add object: {e}")
                            else:
                                logger.warning(f"Skipping document {item['doc_id']} due to embedding failure")

                    # Update progress bar
                    pbar.update(len(batch))

                    # Small delay to avoid overwhelming the server
                    time.sleep(0.1)

            logger.info(f"Successfully imported {success_count} out of {len(chunks)} chunks")
            return success_count
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def query(self, query_text, limit=5, include_vector=False):
        """Query the Weaviate database for relevant content"""
        try:
            # Generate query vector
            query_vector = self.generate_embedding(query_text)

            if not query_vector:
                logger.error("Failed to generate query embedding")
                return []

            # Build GraphQL query
            graphql_fields = ["doc_id", "url", "text", "source", "url_path"]
            if include_vector:
                graphql_fields.append("_additional { vector }")

            # Execute the search
            result = self.client.query.get(
                class_name=self.class_name,
                properties=graphql_fields
            ).with_near_vector({
                "vector": query_vector,
                "certainty": 0.7  # Adjust certainty threshold as needed
            }).with_limit(limit).do()

            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                return result["data"]["Get"][self.class_name]
            else:
                logger.warning(f"Unexpected query result structure: {result}")
                return []
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_stats(self):
        """Get statistics about the Weaviate database"""
        try:
            stats = {
                "object_count": 0,
                "class_properties": []
            }

            # Get object count
            result = self.client.query.aggregate(self.class_name).with_meta_count().do()
            if "data" in result and "Aggregate" in result["data"] and self.class_name in result["data"]["Aggregate"]:
                stats["object_count"] = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]

            # Get schema info
            schema = self.client.schema.get(self.class_name)
            if schema and "properties" in schema:
                stats["class_properties"] = [prop["name"] for prop in schema["properties"]]

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


def main():
    # Initialize the Weaviate DB handler
    db = WaldenWeaviateDB(
        data_folder="walden_data",
        weaviate_url="http://localhost:8080",  # Change to your Weaviate instance URL
        api_key=None  # Add API key if needed
    )

    # Create schema
    print("Creating Weaviate schema...")
    db.create_schema()

    # Load data
    print("Loading data into Weaviate...")
    count = db.load_data(batch_size=50)
    print(f"Loaded {count} documents into Weaviate")

    # Get stats
    stats = db.get_stats()
    print(f"Weaviate DB stats: {stats}")

    # Test query
    print("\nTesting search functionality...")
    test_query = "What programs does Walden University offer?"
    results = db.query(test_query, limit=3)

    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"URL: {result['url']}")
        print(f"Text preview: {result['text'][:150]}...")


if __name__ == "__main__":
    main()