import numpy as np
from body import body
import simulation
import matplotlib.pyplot as plt

G=1
mH = 1
mL = 1e-6
body11 = body(mH, (0.0, 0.0), (0.0, 0.8)) 
body21 = body(mL, (1.0, 0.0), (0.0, 0.0)) 
bodies = [body11, body21]

print(f"Preliminary heavy velocity: {body11.velocity}")
print(f"Preliminary light velocity: {body21.velocity}")

simulation.toCenterOfMassFrame(bodies)
Rcom, Vcom = simulation.centerOfMass(bodies)
print(f"Initial momentum: {np.linalg.norm(simulation.momentum(bodies))}")
print(f"Transformed heavy velocity: {body11.velocity}")
print(f"Transformed light velocity: {body21.velocity}")
r1 = body11.position
r2 = body21.position

v1 = body11.velocity
v2 = body21.velocity



rRel = r2 - r1
vRel = v2 - v1
v0 = np.linalg.norm(vRel)
d = 1


aRel = 1/ ((2/d) - (v0**2 / (G * (mH + mL)))) 
T = 2*np.pi * np.sqrt(aRel**3 / (G * (mH + mL)))

tf = 11 * T
ti = 0
nSteps = 10000
simulation.Verlet(bodies, tf, ti, nSteps)
t = np.linspace(ti, tf, nSteps+1)
r1 = np.array(body11.posHistory)
r2 = np.array(body21.posHistory)
v1 = np.array(body11.velHistory)
v2 = np.array(body21.velHistory)

plt.plot(r1[:, 0], r1[:, 1], label = 'Body 1')
plt.plot(r2[:, 0], r2[:, 1], label = 'Body 2')
plt.scatter(r1[0, 0], r1[0, 1], marker="x", label="Body 1 initial")
plt.scatter(r2[0, 0], r2[0, 1], marker="x", label="Body 2 initial")

plt.scatter(Rcom[0], Rcom[1], color="black", label="Center of mass")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.savefig("Trajectories1.png")
plt.show()

K1 = 0.5 * mH * np.sum(v1**2, axis = 1)
K2 = 0.5 * mL * np.sum(v2**2, axis = 1)
separation = np.linalg.norm(r2 - r1, axis = 1)
U1 = -0.5 * G * mH * mL / separation
U2 = -0.5 * G * mH * mL / separation
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
plt.savefig("Energies1.png")
plt.show()



radius1 = np.linalg.norm(r1 - Rcom, axis=1)
radius2 = np.linalg.norm(r2 - Rcom, axis=1)







RcomHistory = (mH*r1 + mL*r2) / (mH+mL)
RcomDev = []
for i in range(len(t)):
        RcomDev.append(np.linalg.norm(RcomHistory[i] - RcomHistory[0]))
plt.figure()
plt.plot(t, RcomDev)
plt.xlabel("Time")
plt.ylabel("Center of Mass Deviation")
plt.tight_layout()
plt.savefig("CenterOfMass1.png")
plt.show()
print("Maximum center of mass deviation:", np.max(RcomDev))


fig, ax = plt.subplots()

ax.plot(r1[:, 0], r1[:, 1], label="Heavy body")
ax.scatter(Rcom[0], Rcom[1], marker="x", label="Center of mass")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Elliptical: Heavy Body Close-up")
ax.set_aspect("equal", adjustable="box")
ax.legend()

plt.tight_layout()
plt.savefig("HeavyCloseupElliptical.png")
plt.show()
# Simulation 2
vc = np.sqrt(G * (mH + mL) / d)
body12 = body(mH, (0.0, 0.0), (0.0, 0.0)) 
body22 = body(mL, (1.0, 0.0), (0.0, vc)) 
bodies = [body12, body22]

simulation.toCenterOfMassFrame(bodies)
Rcom, Vcom = simulation.centerOfMass(bodies)
print(f"Initial momentum: {np.linalg.norm(simulation.momentum(bodies))}")

r1 = body12.position
r2 = body22.position

v1 = body12.velocity
v2 = body22.velocity

rRel = r2 - r1
vRel = v2 - v1
v0 = np.linalg.norm(vRel)
d = 1

aRel = 1/ ((2/d) - (v0**2 / (G * (mH + mL)))) 
T = 2*np.pi * d / vc

tf = 11 * T
ti = 0
nSteps = 10000
simulation.Verlet(bodies, tf, ti, nSteps)
t = np.linspace(ti, tf, nSteps+1)
r1 = np.array(body12.posHistory)
r2 = np.array(body22.posHistory)
v1 = np.array(body12.velHistory)
v2 = np.array(body22.velHistory)

plt.plot(r1[:, 0], r1[:, 1], label = 'Body 1')
plt.plot(r2[:, 0], r2[:, 1], label = 'Body 2')
plt.scatter(r1[0, 0], r1[0, 1], marker="x", label="Body 1 initial")
plt.scatter(r2[0, 0], r2[0, 1], marker="x", label="Body 2 initial")

plt.scatter(Rcom[0], Rcom[1], color="black", label="Center of mass")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.savefig("Trajectories2.png")
plt.show()

K1 = 0.5 * mH * np.sum(v1**2, axis = 1)
K2 = 0.5 * mL * np.sum(v2**2, axis = 1)
separation = np.linalg.norm(r2 - r1, axis = 1)
U1 = -0.5 * G * mH * mL / separation
U2 = -0.5 * G * mH * mL / separation
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
plt.savefig("Energies2.png")
plt.show()



radius1 = np.linalg.norm(r1 - Rcom, axis=1)
radius2 = np.linalg.norm(r2 - Rcom, axis=1)






RcomHistory = (mH*r1 + mL*r2) / (mH+mL)
RcomDev = []
for i in range(len(t)):
        RcomDev.append(np.linalg.norm(RcomHistory[i] - RcomHistory[0]))
plt.figure()
plt.plot(t, RcomDev)
plt.xlabel("Time")
plt.ylabel("Center of Mass Deviation")
plt.tight_layout()
plt.savefig("CenterOfMass2.png")
plt.show()
print("Maximum center of mass deviation:", np.max(RcomDev))


fig, ax = plt.subplots()
ax.plot(r1[:,0], r1[:,1], label="Heavy body (circular)")
ax.scatter(Rcom[0], Rcom[1], marker="x", label="Center of mass")
ax.set(xlabel="x", ylabel="y", title="Circular: heavy-body close-up")
ax.set_aspect("equal", adjustable="box")
ax.legend()
fig.savefig("HeavyCloseupCircular.png")
plt.close(fig)


import A6Simulation
state = (1, 0, 0, 1, 0, 0)
ti = 0
Tc = 2 * np.pi
tf = np.pi * 2 * 11
nSteps = 10000
state = A6Simulation.gravAccel(state)
states = A6Simulation.Verlet(state, tf, ti, nSteps)


x = []
y = []
h = (tf - ti) / nSteps
for i, state in enumerate(states):
    x.append(state[0])
    y.append(state[1])
relative = r2-r1
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")
ax.plot(relative[:,0], relative[:,1], label="Relative orbit")
ax.plot(x, y, label="Fixed Center Orbit")
ax.plot(0, 0, "o", label="Central mass")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
fig.savefig("FixedCenterComparison.png")
