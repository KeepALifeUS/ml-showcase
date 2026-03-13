"""
Elliott Wave Pattern Recognition Module.

Comprehensive pattern detection for all Elliott Wave structures with
for high-accuracy recognition in crypto markets.
"""

from .impulse_wave import ImpulseWaveDetector, ImpulseWave
from .corrective_wave import CorrectiveWaveDetector, CorrectiveWave
from .diagonal_wave import DiagonalWaveDetector, DiagonalWave
from .triangle_wave import TriangleWaveDetector, TriangleWave
from .flat_wave import FlatWaveDetector, FlatWave
from .zigzag_wave import ZigzagWaveDetector, ZigzagWave

__all__ = [
    # Impulse Patterns
    'ImpulseWaveDetector',
    'ImpulseWave',
    
    # Corrective Patterns
    'CorrectiveWaveDetector', 
    'CorrectiveWave',
    
    # Diagonal Patterns
    'DiagonalWaveDetector',
    'DiagonalWave',
    
    # Triangle Patterns
    'TriangleWaveDetector',
    'TriangleWave',
    
    # Flat Patterns
    'FlatWaveDetector',
    'FlatWave',
    
    # Zigzag Patterns
    'ZigzagWaveDetector',
    'ZigzagWave',
]