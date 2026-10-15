import numpy as np
G = 1
M=1
m=1
def mag(state: tuple[float, float, float, float, float, float]):
    return np.sqrt(state[0]**2 + state[1]**2)

def gravAccel(state: tuple[float, float, float, float, float, float]):
    ax = -(G * M / mag(state)**3) * state[0] 
    ay = -(G * M / mag(state)**3) * state[1]
    return (state[0], state[1], state[2], state[3], ax, ay)

def Verlet(state, tf, ti, nSteps):
    h = (tf - ti) / nSteps
    states = [state]
    for i in range(nSteps):
        rNextX = state[0] + state[2] * h + 0.5 * h**2 * state[4]
        rNextY = state[1] + state[3] * h + 0.5 * h**2 * state[5]
        aNext = gravAccel((rNextX, rNextY, state[2], state[3], state[4], state[5]))
        vNextX = state[2] + 0.5 * h * (state[4] + aNext[4])
        vNextY = state[3] + 0.5 * h * (state[5] + aNext[5])
        state = (rNextX, rNextY,vNextX, vNextY,aNext[4], aNext[5])
        states.append(state)
    return states


def energy(state):
    v = np.sqrt(state[2]**2 + state[3]**2)
    r = mag(state)
    K = 0.5 * m * v**2
    U = -(G * M * m) / r
    E = K + U
    return K, U, E


def radius(state):
    return mag(state)


def speed(state):
    return np.sqrt(state[2]**2 + state[3]**2)


