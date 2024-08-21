import numpy as np
import matplotlib.pyplot as plt

# Generate values for the x-axis
t = np.linspace(0, 1/5 * np.pi, 1000)

# Compute the sine of x
f=10;
Angular_frequency=10*2*np.pi
y = np.sin(Angular_frequency*t)
# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Sine Wave')
plt.title('Sine Function')
plt.xlabel('time (s)')
plt.ylabel('y(t) (m)')
plt.grid(True)
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()