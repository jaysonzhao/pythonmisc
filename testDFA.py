from dfa import dfa
from generate import power_law_noise

true_exp = 0.8
x = power_law_noise(2**12, true_exp)
print(x)
scales, fluct, alpha = dfa(x)
print("True scaling exponent: {}".format(true_exp))
print("Estimated DFA exponent: {}".format(alpha))