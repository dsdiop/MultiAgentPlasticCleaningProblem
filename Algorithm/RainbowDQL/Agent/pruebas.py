import numpy as np
import matplotlib.pyplot as plt

def anneal_nu(p, p1=[0., 1], p2=[0.5, 0.5], p3=[0.5, 0.], p4=[1., 1.]):
    if p < p2[0]:
        first_p = p1
        second_p = p2
    elif p < p3[0]:
        first_p = p2
        second_p = p3
    elif p < p4[0]:
        first_p = p3
        second_p = p4

    return (second_p[1] - first_p[1]) / (second_p[0] - first_p[0]) * (p - first_p[0]) + first_p[1]

eps = 2000
p = np.arange(eps)/eps
listapr = []
for i in p:
    listapr.append(anneal_nu(i))
plt.plot(listapr)
plt.show()