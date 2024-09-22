import _script_startup
import torch
import matplotlib.pyplot as plt

from src.rgrav import ChebyshevMagicNumbers

torch.set_default_dtype(torch.float64)


alpha_arr = torch.logspace(-8, -0.3, 128, dtype=torch.float64)
it_arr = []
for alpha in alpha_arr:
    cmn = ChebyshevMagicNumbers(alpha.item())
    for n in range(2, 10000):
        a_n = cmn.a(n)
        if a_n != a_n:
            it_arr.append(n)
            break
plt.plot(alpha_arr, (2/3)*torch.tensor(it_arr))
plt.show()
