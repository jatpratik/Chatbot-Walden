import json
import os
import re
import tiktoken
from typing import List, Dict, Any
from tqdm import tqdm


class WaldenDataProcessor:
    def __init__(self, output_folder="walden_data", chunk_size=500, chunk_overlap=100):
        """
        Initialize the data processor.

        Args:
            output_folder: Folder to store processed data
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def extract_json_documents(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON documents from the input text."""
        # Find all JSON objects in the text
        docs = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{"doc_id":'):
                try:
                    doc = json.loads(line)
                    docs.append(doc)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON line: {line[:50]}...")

        return docs

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        # First split by paragraphs
        paragraphs = re.split(r'\n\n+', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If paragraph is too large, split it into sentences
            if para_tokens > self.chunk_size:
                sentences = re.split(r'(?<=[.!?]) +', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_tokens = self.count_tokens(sentence)

                    # If adding this sentence would exceed chunk size
                    if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                        chunks.append(current_chunk)

                        # Create overlap with previous chunk
                        if self.chunk_overlap > 0:
                            # Find a good breakpoint for overlap
                            tokens = self.tokenizer.encode(current_chunk)
                            if len(tokens) > self.chunk_overlap:
                                overlap_tokens = tokens[-self.chunk_overlap:]
                                overlap_text = self.tokenizer.decode(overlap_tokens)
                                current_chunk = overlap_text
                                current_tokens = len(overlap_tokens)
                            else:
                                current_chunk = current_chunk
                                current_tokens = current_tokens
                        else:
                            current_chunk = ""
                            current_tokens = 0

                    # Add sentence to current chunk
                    separator = " " if current_chunk else ""
                    current_chunk += separator + sentence
                    current_tokens += sentence_tokens
            else:
                # If adding this paragraph would exceed chunk size
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)

                    # Create overlap with previous chunk
                    if self.chunk_overlap > 0:
                        # Find a good breakpoint for overlap
                        tokens = self.tokenizer.encode(current_chunk)
                        if len(tokens) > self.chunk_overlap:
                            overlap_tokens = tokens[-self.chunk_overlap:]
                            overlap_text = self.tokenizer.decode(overlap_tokens)
                            current_chunk = overlap_text
                            current_tokens = len(overlap_tokens)
                        else:
                            current_chunk = current_chunk
                            current_tokens = current_tokens
                    else:
                        current_chunk = ""
                        current_tokens = 0

                # Add paragraph to current chunk
                separator = "\n\n" if current_chunk else ""
                current_chunk += separator + para
                current_tokens += para_tokens

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single document into chunks with metadata."""
        doc_id = doc.get("doc_id", "unknown")
        url = doc.get("url", "")
        source = doc.get("source", "")
        url_path = doc.get("url_path", "")
        text = doc.get("text", "")

        # Split text into chunks
        chunks = self.chunk_text(text)

        # Create chunk documents
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                "doc_id": f"{doc_id}-chunk-{i + 1}",
                "url": url,
                "text": chunk_text,
                "source": source,
                "url_path": url_path,
                "parent_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunk_docs.append(chunk_doc)

        return chunk_docs

    def process_input_text(self, input_text: str) -> List[Dict[str, Any]]:
        """Process raw input text into chunked documents."""
        # Extract documents from input text
        documents = self.extract_json_documents(input_text)
        print(f"Extracted {len(documents)} documents from input")

        # Process each document
        all_chunks = []
        for doc in tqdm(documents, desc="Processing documents"):
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], filename: str = "rag_chunks.jsonl") -> str:
        """Save chunks to a JSONL file in the output folder."""
        output_path = os.path.join(self.output_folder, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Saved {len(chunks)} chunks to {output_path}")
        return output_path


def main(input_text: str, output_folder: str = "walden_data"):
    """Process input text and prepare it for Weaviate."""
    processor = WaldenDataProcessor(
        output_folder=output_folder,
        chunk_size=350,  # Adjust based on your needs
        chunk_overlap=50  # Adjust based on your needs
    )

    # Process and chunk the documents
    chunks = processor.process_input_text(input_text)

    # Save to JSONL file
    output_path = processor.save_chunks(chunks)

    print(f"\nData preparation complete. Files saved to {output_folder}")
    print(f"You can now run the Weaviate loader script with this data")

    # Show sample of processed data
    if chunks:
        print("\nSample chunk:")
        sample = chunks[0]
        print(f"ID: {sample['doc_id']}")
        print(f"URL: {sample['url']}")
        print(f"Token count: {processor.count_tokens(sample['text'])}")
        print(f"Text preview: {sample['text'][:150]}...")

    return output_path


if __name__ == "__main__":
    # You would replace this with reading from your file
    with open('walden_data/data.txt', 'r', encoding='utf-8') as f:
        input_text = f.read()

    main(input_text)