"""
API Layer for Elliott Wave Analyzer.

RESTful API, WebSocket server, and GraphQL endpoints with
, authentication, and real-time capabilities.
"""

from .rest_api import ElliottWaveAPI, app
from .websocket_server import WebSocketServer, ConnectionManager
from .graphql_schema import GraphQLSchema, WaveQuery, WaveMutation
from .authentication import AuthManager, JWTAuth, APIKeyAuth

__all__ = [
    # REST API
    'ElliottWaveAPI',
    'app',
    
    # WebSocket
    'WebSocketServer',
    'ConnectionManager',
    
    # GraphQL
    'GraphQLSchema',
    'WaveQuery',
    'WaveMutation',
    
    # Authentication
    'AuthManager',
    'JWTAuth',
    'APIKeyAuth',
]