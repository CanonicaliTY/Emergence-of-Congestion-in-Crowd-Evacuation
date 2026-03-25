import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("exit_times_200_agents.csv")
df.index = np.linspace(1, 6, 30)

mean = np.array([np.mean(df[f"Run {i+1}"], axis = 1) for i in range(10)])
std = np.array([np.std(df[f"Run {i+1}"], axis = 1) for i in range(10)])

'''
plt.plot(df["Run 1"], label="Run 1")
plt.plot(df["Run 2"], label="Run 2")
plt.plot(df["Run 3"], label="Run 3")
plt.plot(df["Run 4"], label="Run 4")
plt.plot(df["Run 5"], label="Run 5")
plt.plot(df["Run 6"], label="Run 6")
plt.plot(df["Run 7"], label="Run 7")
plt.plot(df["Run 8"], label="Run 8")
plt.plot(df["Run 9"], label="Run 9")
plt.plot(df["Run 10"], label="Run 10")
'''
plt.plot(mean, label="Mean")
plt.fill_between(df.index, mean - std, mean + std, alpha=0.2)
plt.legend()

plt.xlabel("Desired Speed (m/s)")
plt.ylabel("Exit Time (s)")
plt.title("Exit Time vs Desired Speed")
plt.savefig("exit_times_200_agents.jpg",dpi=600)
plt.show()