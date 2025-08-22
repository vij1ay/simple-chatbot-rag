"""
1. Place any **PDF files** you want to use in the `data` folder.

2. Run the populate script:

   ```bash
   python populate_data.py
   ```

   * This processes all PDFs in `data/`
   * Stores embeddings and indexes in the `chroma/` folder

3. Ask questions from your data:

   ```bash
   python query_data.py --query "What is XYZ COMPANY GROSS PROFIT in 2002?"
   ```

   * The system searches embeddings in Chroma
   * Uses the local LLM to answer based on context


"""

import argparse
import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from embedding_helper import get_embedding_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHROMA_DB_PATH = "chroma"
FILE_DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logger.info("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    # for document in documents:
    #     print(document, "\n\n")
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(FILE_DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents chunks in DB: {len(existing_ids)}")

    # Only add documents which the chunk don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        logger.info(f"Adding new documents chunks: {len(new_chunks)}.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        logger.info("No new documents chunks to add")


def calculate_chunk_ids(chunks):
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # increment the index. If the page ID is the same as the last one,
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Get the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)


if __name__ == "__main__":
    main()
