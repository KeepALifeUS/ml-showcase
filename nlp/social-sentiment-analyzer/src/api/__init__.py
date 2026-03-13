"""
API module for system analysis sentiments

Contains REST API, GraphQL API and WebSocket server for
integration with external systems.
"""

from .rest_api import SocialSentimentAPI
from .graphql_api import GraphQLAPI  
from .websocket_server import WebSocketServer
from .authentication import AuthManager

__all__ = [
    "SocialSentimentAPI",
    "GraphQLAPI",
    "WebSocketServer", 
    "AuthManager",
]