"""FastAPI server for the RAG-based chatbot.

This module provides a web server implementation using FastAPI to serve the chatbot interface
and handle chat interactions. It includes routes for serving the chat UI and processing
chat messages through the RAG system.

The server:
- Serves a static HTML chat interface
- Processes chat messages through the RAG graph
- Handles conversation state and thread management
- Returns AI-generated responses with context from the document collection
"""

import uvicorn
import webbrowser
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rag_graph import ChatbotRagRetrieval

from config import DEFAULT_PORT as webserver_port

def create_app(graph: ChatbotRagRetrieval) -> FastAPI:
    """Create and configure the FastAPI application.
    
    This function sets up the FastAPI application with all necessary routes and middleware.
    It configures static file serving for the chat UI and establishes the chat API endpoint.
    
    Args:
        graph (ChatbotRagRetrieval): Instance of the RAG chatbot graph for processing messages
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="RAG Chatbot API",
        description="API for interacting with the RAG-based chatbot",
        version="1.0.0"
    )
    
    # Serve static files from assets directory
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_chat() -> HTMLResponse:
        """Serve the chat interface HTML page.
        
        Returns:
            HTMLResponse: The rendered chat interface HTML
        """
        with open("assets/chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    @app.post("/api/chat")
    async def chat_endpoint(request: Request) -> Dict[str, str]:
        """Process chat messages and return AI responses.
        
        This endpoint:
        1. Receives the user message and thread ID
        2. Passes the message through the RAG graph
        3. Logs any tool usage
        4. Returns the final response
        
        Args:
            request (Request): FastAPI request object containing chat message data
        
        Returns:
            Dict[str, str]: Response containing the AI's message
                {
                    "response": str  # The AI's response text
                }
        """
        data = await request.json()
        print("data >> ", data)
        user_message = data.get("message", "")
        thread_id = data.get("thread_id", "-1")
        
        # Configure and invoke the graph
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )
        
        # # Log tool calls for debugging
        # for msg in result["messages"]:
        #     if hasattr(msg, "tool_calls"):
        #         for tool_call in msg.tool_calls:
        #             print("Tool >> ", tool_call.get("name"), tool_call.get("args"))
        
        final_response = result["messages"][-1].content
        return {"response": final_response}
    
    return app

def main() -> None:
    """Initialize and start the chatbot server.
    
    This function:
    1. Creates the RAG chatbot instance
    2. Initializes the FastAPI application
    3. Opens the chat interface in the default web browser
    4. Starts the uvicorn server
    """
    # Initialize the RAG chatbot
    graph = ChatbotRagRetrieval()
    
    # Create the FastAPI application
    fastapi_app = create_app(graph)
    
    # Open chat interface in browser
    webbrowser.open(f'http://localhost:{webserver_port}')
    
    # Start the server
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=webserver_port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
