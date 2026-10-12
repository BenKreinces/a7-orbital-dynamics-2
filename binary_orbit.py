import numpy as np
from body import body
import simulation
import matplotlib.pyplot as plt
G=1
m1 = 1
m2 = 2
body1 = body(m1, (0.0, 0.0), (0.0, -1.4)) 
body2 = body(m2, (1.0, 0.0), (0.0, 0.0)) 
bodies = [body1, body2]

simulation.toCenterOfMassFrame(bodies)
Rcom, Vcom = simulation.centerOfMass(bodies)
print(f"Initial momentum: {np.linalg.norm(simulation.momentum(bodies))}")



r1 = body1.position
r2 = body2.position

v1 = body1.velocity
v2 = body2.velocity

rRel = r2 - r1
vRel = v2 - v1
v0 = np.linalg.norm(vRel)
d = np.linalg.norm(rRel)


aRel = 1/ ((2/d) - (v0**2 / (G * (m1 + m2)))) 
T = 2*np.pi * np.sqrt(aRel**3 / (G * (m1 + m2)))

tf = 11 * T
ti = 0
nSteps = 10000
simulation.Verlet(bodies, tf, ti, nSteps)
t = np.linspace(ti, tf, nSteps+1)
r1 = np.array(body1.posHistory)
r2 = np.array(body2.posHistory)
v1 = np.array(body1.velHistory)
v2 = np.array(body2.velHistory)

plt.plot(r1[:, 0], r1[:, 1], label = 'Body 1')
plt.plot(r2[:, 0], r2[:, 1], label = 'Body 2')
plt.scatter(r1[0, 0], r1[0, 1], marker="x", label="Body 1 initial")
plt.scatter(r2[0, 0], r2[0, 1], marker="x", label="Body 2 initial")

plt.scatter(Rcom[0], Rcom[1], color="black", label="Center of mass")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.savefig("Trajectories.png")
plt.show()

K1 = 0.5 * m1 * np.sum(v1**2, axis = 1)
K2 = 0.5 * m2 * np.sum(v2**2, axis = 1)
separation = np.linalg.norm(r2 - r1, axis = 1)
U1 = -0.5 * G * m1 * m2 / separation
U2 = -0.5 * G * m1 * m2 / separation
E1 = K1 + U1
E2 = K2 + U2
K = K1 + K2
U = U1 + U2
E = E1 + E2

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(t, K1, label = 'Body 1')
axes[0].plot(t, K2, label = 'Body 2')
axes[0].plot(t, K, label = 'Total')
axes[0].set_ylabel('Kinetic Energy')
axes[1].plot(t, U1, label = 'Body 1')
axes[1].plot(t, U2, label = 'Body 2')
axes[1].plot(t, U, label = 'Total')
axes[1].set_ylabel('Potential Energy')
axes[2].plot(t, E1, label = 'Body 1')
axes[2].plot(t, E2, label = 'Body 2')
axes[2].plot(t, E, label = 'Total')
axes[2].set_ylabel('Total Energy')
axes[0].legend()
axes[2].legend()
axes[1].legend()
axes[2].set_xlabel("Time")
axes[1].set_xlabel("Time")
axes[0].set_xlabel("Time")
plt.tight_layout()
plt.savefig("Energies.png")
plt.show()



radius1 = np.linalg.norm(r1 - Rcom, axis=1)
radius2 = np.linalg.norm(r2 - Rcom, axis=1)




def measurePeriod(radius, t):
    min = []
    periapsis = []
    apoapsis = []
    for i in range(1, len(radius)-1):
        if radius[i] < radius[i-1] and radius[i] <= radius[i+1]:
            min.append(t[i])
            periapsis.append(radius[i])
        if radius[i] > radius[i-1] and radius[i] >= radius[i+1]:
            apoapsis.append(radius[i])
    periods = []
    for i in range(len(min)-1):
        periods.append(min[i+1] - min[i])
    return np.mean(periods), np.mean(periapsis), np.mean(apoapsis)
T1, per1, apo1 = measurePeriod(radius1, t)
T2, per2, apo2 = measurePeriod(radius2, t)

a1 = (per1 + apo1) / 2
a2 = (per2 + apo2) / 2
e1 = (apo1 - per1) / (per1 + apo1)
e2 = (apo2 - per2) / (per2 + apo2)

print(f"Body 1 period: {T1}")
print(f"Body 2 period:{T2}")
print(f"Expected Value:{T}")
print(f"Difference between Body 1 period and expected value: {T - T1}")
print(f"Difference between Body 2 period and expected value: {T - T2}")
print(f"a1: {a1}, a2: {a2}, e1: {e1}, e2: {e2}")
print(f"a1/a2: {a1 / a2}, m2/m1: {m2/m1}")



RcomHistory = (m1*r1 + m2*r2) / (m1+m2)
RcomDev = []
for i in range(len(t)):
        RcomDev.append(np.linalg.norm(RcomHistory[i] - RcomHistory[0]))
plt.figure()
plt.plot(t, RcomDev)
plt.xlabel("Time")
plt.ylabel("Center of Mass Deviation")
plt.tight_layout()
plt.savefig("CenterOfMass.png")
plt.show()
print("Maximum center of mass deviation:", np.max(RcomDev))