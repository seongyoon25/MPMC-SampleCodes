import matplotlib.pyplot as plt
import numpy as np


i = ...
j = ...
plt.figure(figsize=(5, 4))

# Observation
plt.plot(..., 'k--')
plt.plot(..., 'ko', markersize=10)
plt.plot([], [], 'k--o', label='Observation')

# Input
plt.plot(..., 'b*', label='Input')

# Prediction
plt.plot(..., 'r')
plt.plot(..., 'r^', markersize=10)
plt.plot([], [], 'r-^', label='Prediction', markersize=10)

# 95% CI
x_lower = ...
y_lower = ...
x_upper = ...
y_upper = ...
plt.fill(np.concatenate([x_lower, x_upper[::-1]]), 
         np.concatenate([y_lower, y_upper[::-1]]),
         edgecolor='orange', facecolor='yellow', label='95% CI')

# Configure
plt.grid()
plt.ylabel('SOH (%)', size=16)
plt.yticks(np.arange(0.8, 1+0.04, 0.04),
           np.int32(np.arange(0.8, 1+0.04, 0.04)*100), fontsize=12)
plt.ylim([0.79, 1.01])
plt.xlabel('Cycle number', size=16)
plt.xticks(fontsize=12)
plt.legend(fontsize=12, loc='lower left')

# Save and Show
figure_name = ...
plt.tight_layout()
plt.savefig(f'{figure_name}.eps', bbox_inches='tight', dpi=300)
plt.show()
