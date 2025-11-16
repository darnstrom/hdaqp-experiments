import numpy as np
import matplotlib.pyplot as plt

bins = np.logspace(-11,6,17)
bins = np.append(bins,1e100)

plt.subplot(1,2,1)
data = np.loadtxt("build/tests/daqp_diff_lexls.dat", delimiter=None)
plt.hist(-data[data <0], bins=bins, edgecolor='black', label="lexls worse")
plt.hist(data[data >0], bins=bins, edgecolor='black', label="DAQP worse", alpha=0.5)
plt.xscale('log')
plt.title("DAQP vs lexls")
plt.ylabel("Frequency")
plt.xlabel("Difference")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([1e-11,1e5])
plt.legend()

plt.subplot(1,2,2)
data = np.loadtxt("build/tests/daqp_diff_nipm.dat", delimiter=None)
plt.hist(-data[data <0], bins=bins, edgecolor='black', label="NIPM worse")
plt.hist(data[data >0], bins=bins, edgecolor='black', label="DAQP worse", alpha=0.5)
plt.title("DAQP vs NIPM")
plt.ylabel("Frequency")
plt.xlabel("Difference")
plt.xscale('log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([1e-11,1e5])
plt.legend()
plt.tight_layout()

plt.show()
