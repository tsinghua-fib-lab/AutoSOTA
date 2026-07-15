
import numpy as np
import scipy.stats as stats

# Stats from Phi-3.5 run
mu_v = 0.4956
std_v = 0.1015
mu_i = 0.2392
std_i = 0.0556

# Cohen's d Recalc
pooled_std = np.sqrt((std_v**2 + std_i**2) / 2) # Approximation for equal N
d = (mu_v - mu_i) / pooled_std

print(f"Stats:")
print(f"Valid:   mu={mu_v}, std={std_v}")
print(f"Invalid: mu={mu_i}, std={std_i}")
print(f"Pooled Std: {pooled_std:.4f}")
print(f"Calculated d: {d:.4f}")

# Calculate Intersection Point (where pdf1 = pdf2)
# Solve for x: (x-u1)^2/2s1^2 + ln(s1) = (x-u2)^2/2s2^2 + ln(s2)
# This is a quadratic equation.

a = 1/(2*std_v**2) - 1/(2*std_i**2)
b = mu_i/(std_i**2) - mu_v/(std_v**2)
c = mu_v**2/(2*std_v**2) - mu_i**2/(2*std_i**2) - np.log(std_i/std_v)

roots = np.roots([a, b, c])
print(f"Intersection points: {roots}")

intersection = [r for r in roots if mu_i < r < mu_v]
if not intersection:
    intersection = float(roots[0]) if abs(roots[0]-mu_i) < abs(roots[0]-mu_v) else float(roots[1])
else:
    intersection = intersection[0]

# Calculate Overlap Area
# Area = Area under left tail of Valid + Area under right tail of Invalid (if Valid is on right)
# Wait, if V is right (0.49 > 0.23):
# Overlap = P(Valid < Intersection) + P(Invalid > Intersection)

cdf_v_at_inter = stats.norm.cdf(intersection, mu_v, std_v)
cdf_i_at_inter = stats.norm.cdf(intersection, mu_i, std_i)

overlap_area = cdf_v_at_inter + (1 - cdf_i_at_inter)

print(f"\nIntersection X: {intersection:.4f}")
print(f"Invalid > X (False Positive Rate): {(1-cdf_i_at_inter)*100:.2f}%")
print(f"Valid < X   (False Negative Rate): {cdf_v_at_inter*100:.2f}%")
print(f"Total Area of Overlap: {overlap_area:.4f}")
