import numpy as np

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
    h = (tf - ti) / nSteps
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
                
            


def energy(bodies):
    K = 0
    U = 0
    for i in range(len(bodies)):
        K += 0.5 * bodies[i].mass * np.dot(bodies[i].velocity, bodies[i].velocity)
        for j in range(i+1, len(bodies)):
            r = np.linalg.norm(bodies[j].position - bodies[i].position)
            U += -1 * G * bodies[j].mass * bodies[i].mass / r
    E = K+U
    
    return K, U, E


def momentum(bodies):
    p = np.array([0.0, 0.0])
    for body in bodies:
        p += body.velocity * body.mass
    return p


def centerOfMass(bodies):
    M = 0
    R = np.array([0.0, 0.0])
    V = np.array([0.0, 0.0])
    for body in bodies:
        R += body.mass * body.position
        V += body.mass * body.velocity
        M += body.mass
    return R/M, V/M

def toCenterOfMassFrame(bodies):
    rCM, vCM = centerOfMass(bodies)
    for body in bodies:
        body.position -= rCM
        body.velocity -= vCM
        


