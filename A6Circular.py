import simulation
import matplotlib.pyplot as plt
import numpy as np
state = (1, 0, 0, 1, 0, 0)
ti = 0
Tc = 2 * np.pi
tf = np.pi * 2 * 11
nSteps = 10000
state = simulation.gravAccel(state)
states = simulation.Verlet(state, tf, ti, nSteps)


x = []
y = []
t = []
K = []
U = []
E = []

h = (tf - ti) / nSteps
for i, state in enumerate(states):
    x.append(state[0])
    y.append(state[1])
    t.append(ti + i * h)
    energies = simulation.energy(state)
    K.append(energies[0])
    U.append(energies[1])
    E.append(energies[2])

fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")
ax.plot(x, y, label="Orbit")
ax.plot(0, 0, "o", label="Central mass")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()

plt.plot(t, x, label="x(t)")
plt.plot(t, y, label="y(t)")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()

plt.plot(t, K, label="K")
plt.plot(t, U, label="U")
plt.plot(t, E, label="E")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()

#Take the time when the orbit crosses x axis, then average each of the measured periods
periods = []
for i in range(1, len(y)):
    if y[i - 1] < 0 and y[i] >= 0 and x[i] > 0:
        periods.append(t[i])

measuredPeriods = []
for i in range(1, len(periods)):
    measuredPeriods.append(periods[i] - periods[i - 1])

measuredPeriod = np.mean(measuredPeriods)
err = abs(measuredPeriod - Tc) / Tc * 100

print(f"Expected period: {Tc}")
print(f"Measured period: {measuredPeriod}")
print(f"Fractional error: {err}%")
