from PatchFormula import RMA
import numpy as np
import random as rd
crossoverRate = 0.8
mutationRate = 0.2
populationSize = 6
chomosomes = np.array([])
def initialise(i):
    for i in range(populationSize):
        width_mm = rd.uniform(25,50)
        length_mm = rd.uniform(25,45)
        thickness_mm = rd.uniform(1,5)
        chomosomes = np.append(chomosomes, RMA(width_mm, length_mm,thickness_mm))

fitness = np.array([chomosome.volume_mm3 for chomosome in chomosomes])
def elitism(fitness) -> float:
    return np.argmin(fitness),min(fitness)
# print(fitness)
# print (elitism(fitness))