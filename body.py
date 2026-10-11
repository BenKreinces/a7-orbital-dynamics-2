from dataclasses import dataclass
from typing import TypeAlias, Dict, Union, List, Optional
import numpy as np
@dataclass

class body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.posHistory = []
        self.velHistory = []

    def record(self):
        self.posHistory.append(self.position.copy())
        self.velHistory.append(self.position.copy())



        
