# %%
import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
x = np.linspace(0, 100, 500)
L = 99  # Maximum value of y
k = 0.1  # Steepness of the curve
x0 = 50  # Midpoint (x-value where y is half of L)

# Sigmoid function
y = L / (1 + np.exp(-k * (x - x0)))

# Plotting
plt.figure(figsize=(8, 6))
emp = plt.plot(x, y, label="Empiricism", linewidth=3)  # Thick line
goal = plt.plot(
    x,
    [
        L - 25,
    ]
    * len(x),
    label="Goal of this repo",
    linestyle="--",
    color="green",
    linewidth=2,
)
conf = plt.plot(
    x,
    [
        L - 3,
    ]
    * len(x),
    label="Conference Publication Threshold",
    linestyle="--",
    color="orange",
    linewidth=2,
)
journ = plt.plot(
    x,
    [
        L + 0.5,
    ]
    * len(x),
    label="Journal Publication Threshold",
    linestyle="--",
    color="red",
    linewidth=2,
)
plt.xlabel("Fooling Around (%)")
plt.ylabel("Finding Out (%)")
plt.title("Knowledge Creation")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.gca().set_xlim(0, 100)
plt.gca().set_ylim(0, 100)
# plt.show()

plt.savefig("knowledge_creation.png")
# %%
