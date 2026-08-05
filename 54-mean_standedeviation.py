import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- 1. Generate data ----------
np.random.seed(42)
MEAN = 50
N = 600

# Dataset A: Low standard deviation (tight)
low_sd = 5
data_low = np.random.normal(MEAN, low_sd, N)
# print(data_low)
length_of_data_low = len(data_low)
print(length_of_data_low)



# Dataset B: High standard deviation (spread)
high_sd = 6   # change this value to see the effect
data_high = np.random.normal(MEAN, high_sd, N)
print(len(data_high))



# # ---------- 2. Compute mean & standard deviation ----------
# Calculate mean and standrd deviation
mean_low = np.mean(data_low)
sd_low = np.std(data_low)




mean_high = np.mean(data_high)
sd_high = np.std(data_high)


print(f"Low SD  → mean = {mean_low:.2f},  SD = {sd_low:.2f}")
print(f"High SD → mean = {mean_high:.2f},  SD = {sd_high:.2f}")

# ---------- 3. Plot side-by-side histograms ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(data_low, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
ax1.axvline(mean_low, color='red', linestyle='dashed', linewidth=2,
            label=f'mean = {mean_low:.1f}')
ax1.set_title(f'Low SD = {sd_low:.2f}')
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.legend()

ax2.hist(data_high, bins=20, color='#ef4444', alpha=0.7, edgecolor='black')
ax2.axvline(mean_high, color='red', linestyle='dashed', linewidth=2,
            label=f'mean = {mean_high:.1f}')
ax2.set_title(f'High SD = {sd_high:.2f}')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.legend()

plt.suptitle('Same Mean (50) – Different Standard Deviations', fontsize=14)
plt.tight_layout()
plt.show()
#
# ---------- 4. Overlapping density plots ----------
plt.figure(figsize=(10, 5))
plt.hist(data_low, bins=30, density=True, alpha=0.5, color='blue', label=f'Low SD = {sd_low:.2f}')
plt.hist(data_high, bins=30, density=True, alpha=0.5, color='red', label=f'High SD = {sd_high:.2f}')
plt.axvline(MEAN, color='black', linestyle='dashed', linewidth=1, alpha=0.6, label='Mean = 50')
plt.title('Overlapping Distributions: Narrow vs Wide')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# ---------- 5. 68-95-99.7 rule (bell curve) ----------
x = np.linspace(-4, 4, 500)
y = norm.pdf(x, 0, 1)

plt.figure(figsize=(10, 5))
plt.plot(x, y, color='black', linewidth=2.5, label='Bell curve')

# Shade the regions
colors = ['#b3d9ff', '#a3e4a3', '#f7b3b3']
labels = ['68% within ±1σ', '95% within ±2σ', '99.7% within ±3σ']
for k, color in zip([1, 2, 3], colors):
    x_fill = np.linspace(-k, k, 300)
    y_fill = norm.pdf(x_fill, 0, 1)
    plt.fill_between(x_fill, y_fill, color=color, alpha=0.5, label=labels[k-1])

plt.xticks([-3, -2, -1, 0, 1, 2, 3], ['-3σ', '-2σ', '-1σ', 'μ', '+1σ', '+2σ', '+3σ'])
plt.title('The 68-95-99.7 Rule (Empirical Rule)')
plt.xlabel('Standard deviations from the mean')
plt.ylabel('Probability density')
plt.legend()
plt.grid(alpha=0.15)
plt.show()