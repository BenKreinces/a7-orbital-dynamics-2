import numpy as np
import body
from typing import TypeAlias, Dict, Union, List, Optional

G = 1

def gravAccel(bodies):
    accelerations = np.zeros((len(bodies), 2))
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if i != j:
                d = bodies[j].position - bodies[i].position
                r = np.linalg.norm(d)
                accelerations[i] += G * bodies[j].mass * d / r**3
    return accelerations

def Verlet(bodies, tf, ti, nSteps):
    h = (tf - ti)
    for body in bodies:
        body.record()
    for _ in range(nSteps):
        a = gravAccel(bodies)
        for i, body in enumerate(bodies):
            body.position += body.velocity * h + 0.5 * a[i] * h**2
        aNext = gravAccel(bodies)
        for i, body in enumerate(bodies):
            body.velocity += 0.5 * h * (a[i] + aNext[i])
        for body in bodies:
            body.record()
                
            


def energy(state):
    K = 0.5 * state.m * v**2
    U = -(G * M * state.m) / r
    E = K + U
    return K, U, E


def radius(state):
    return mag(state)


def speed(state):
    return np.sqrt(state.vx**2 + state.vy**2)


