"""RAG-based chatbot graph implementation using LangChain and Chroma.

This module implements a conversational retrieval-augmented generation (RAG) system
using LangChain's graph architecture. It combines vector search over a document
collection with a language model to generate contextually relevant responses.
"""

import uvicorn
import webbrowser
from functools import partial, update_wrapper

from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage
)

from utils import get_llm, get_embedding_function
from config import CHROMA_PATH


class ChatbotRagRetrieval:
    """Implements a RAG-based chatbot using LangChain's graph architecture.
    
    This class manages the document retrieval, LLM interaction, and conversation flow
    through a state graph. It uses Chroma as a vector store for document retrieval
    and LangChain's tools for RAG capabilities.

    Attributes:
        embedding_function: Function to create embeddings for vector search
        db: Chroma vector store instance
        llm: Language model instance
        tools: List of RAG tools bound to the LLM
        memory: Conversation memory saver
        graph: Compiled state graph for conversation flow
    """

    def __init__(self):
        """Initialize the RAG chatbot with vector store, LLM, and graph components."""
        # Set up vector store and embedding
        self.embedding_function = get_embedding_function()
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedding_function
        )

        # Initialize LLM and tools
        self.llm = get_llm()
        self.tools = [self.get_rag_retrieve_tool()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Set up memory and graph
        self.memory = MemorySaver()
        self.graph = None
        self.initGraph()

    def invoke(self, *args, **kwargs):
        """Invoke the state graph with the given input state.
        
        Args:
            *args: Variable positional arguments passed to graph.invoke
            **kwargs: Variable keyword arguments passed to graph.invoke
        
        Returns:
            dict: The result state from graph execution
        """
        return self.graph.invoke(*args, **kwargs)

    def get_rag_retrieve_tool(self):
        """Create and return the RAG retrieval tool.
        
        Returns:
            callable: Decorated tool function for RAG retrieval
        """
        @tool(response_format="content_and_artifact")
        def rag_retrieve(query: str):
            """Retrieve information related to a query from vector database.
            
            Args:
                query (str): The search query to find relevant documents
            
            Returns:
                tuple: (serialized_content, retrieved_documents)
                - serialized_content (str): Formatted string of retrieved content
                - retrieved_documents (list): Raw document objects with scores
            """
            try:
                print(f"\nVector Query >> {query}")
                retrieved_docs = self.db.similarity_search_with_score(query, k=2)
                sources = [doc.metadata.get("id", None) for doc, _score in retrieved_docs]
                print("\nDocument Embedding Reffered:")
                for source in sources:
                    print(" > %s" % source)
                print ("\n")
                serialized = "\n\n".join(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                    for doc, _score in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                print("Error in rag_retrieve:", e)
                return "", []
        return rag_retrieve

    
    # @tool(response_format="content_and_artifact")
    def rag_retrieve_logic(self, query: str): # not used. just for calling it directly and test.
        """Retrieve information related to a query from vector database."""
        try:
            print ("\nVector Query >> %s\n" % query)
            retrieved_docs = self.db.similarity_search_with_score(query, k=2)
            print ("retrieved_docs>> ", retrieved_docs)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc, _score in retrieved_docs
            )
            # sources = [doc.metadata.get("id", None) for doc, _score in results]
            return serialized, retrieved_docs
        except Exception as e:
            print ("Error in rag_retrieve:", e)
            return "", []


    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(self, state: MessagesState) -> dict:
        """Generate a tool call for retrieval or direct response.
        
        This node in the graph either generates a tool call to retrieve information
        or provides a direct response if retrieval isn't needed.
        
        Args:
            state (MessagesState): Current conversation state
        
        Returns:
            dict: Updated state with new message
        """
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState) -> dict:
        """Generate a response using retrieved content and conversation context.
        
        This node processes retrieved information and generates a natural response
        incorporating the relevant context.
        
        Args:
            state (MessagesState): Current conversation state with retrieved content
        
        Returns:
            dict: Updated state with generated response
        """
        # Extract recent tool messages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format prompt with retrieved content
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        # print("docs_content >>> ", docs_content)
        
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you cant find the answer from the context, say that you don't know "
            "and tell the user to rephrase the question (dont give any explanation). "
            "Use three sentences maximum and keep the answer concise. "
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}


    def initGraph(self):
        """Initialize and compile the state graph for conversation flow.
        
        Sets up the nodes, edges, and conditions that define how the conversation
        flows between retrieval and response generation.
        """
        graph_builder = StateGraph(MessagesState)

        # Add nodes for the main components
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", ToolNode(self.tools))
        graph_builder.add_node("generate", self.generate)

        # Configure graph structure
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        # Add conditional routing
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )

        # Compile graph with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)

