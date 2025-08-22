"""Utility functions for language model and embedding setup.

This module provides factory functions for initializing language models and embeddings
used by the RAG system. It supports multiple model providers (OpenAI, Google GenAI, Ollama)
and can be configured via environment variables.

Example:
    >>> from utils import get_llm, get_embedding_function
    >>> llm = get_llm()  # Get configured language model
    >>> embeddings = get_embedding_function()  # Get embedding model
"""

import os

from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings

import config

def get_llm() -> BaseChatModel:
    """Initialize and return a language model for chat interactions.
    
    This function sets up a language model based on available credentials and
    configuration. It supports multiple providers:
    
    - OpenAI GPT models
    - Google Gemini models
    - Local Ollama models
    
    The choice of model can be configured through environment variables or
    by modifying the implementation directly.
    
    Configuration Options:
        - OpenAI:
          ```
          # Required: pip install langchain-openai
          from langchain_openai import ChatOpenAI
          llm = ChatOpenAI(model="gpt-4", temperature=0.0)
          # or
          llm = init_chat_model("gpt-4", model_provider="openai", temperature=0.0)
          ```
        
        - Google GenAI:
          ```
          # Required: pip install langchain-google-genai
          from langchain_google_genai import ChatGoogleGenerativeAI
          llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
          # or
          llm = init_chat_model("gemini-pro", model_provider="google_genai", temperature=0.0)
          ```
    
    Returns:
        BaseChatModel: Initialized language model instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    if hasattr(config, "GOOGLE_API_KEY") and config.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Get API key from environment
        google_api_key = config.GOOGLE_API_KEY
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        os.environ["GOOGLE_API_KEY"] = google_api_key        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0
        )
        print ("\n** Initialized Google GenAI LLM **\n")
        return llm
    else: # return local model
        llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0.0)
        print ("\n** Initialized Ollama LLM **\n")
        return llm
    


def get_embedding_function() -> Embeddings:
    """Initialize and return an embedding model for text vectorization.
    
    This function sets up an embedding model for converting text into vector
    representations. It supports multiple providers:
    
    - Ollama (local) embeddings
    - OpenAI embeddings
    - Other LangChain-compatible embedding models
    
    Configuration Options:
        - OpenAI:
          ```
          # Required: pip install langchain-openai
          from langchain_openai import OpenAIEmbeddings
          embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
          ```
        
        - Ollama (default):
          ```
          from langchain_ollama import OllamaEmbeddings
          embeddings = OllamaEmbeddings(model="nomic-embed-text")
          ```
    
    Returns:
        Embeddings: Initialized embedding model instance
    
    Note:
        The default implementation uses Ollama's nomic-embed-text model,
        which runs locally and doesn't require API keys.
    """
    return OllamaEmbeddings(model="nomic-embed-text")
