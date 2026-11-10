from PatchFormula import RMA
import numpy as np
import random as rd
crossoverRate = 0.8
mutationRate = 0.2
populationSize = 6
chomosomes = np.array([])
AntennaBounds_mm = np.array([[25.0, 50.0], [25.0, 45.0], [1.0, 5.0]])

def geometricCheck(antennadesign : RMA) -> bool:
    # Check if the design is within the bounds & W/L >= 1.2
    design = np.array([antennadesign.width_mm, antennadesign.length_mm, antennadesign.thickness_mm])
    if design.shape != 3 or np.all(np.isfinite(design)) == False:
        print("Design dimension is not correct or contains non-finite values.")
        return False
    return bool( 
        np.all(design) >= AntennaBounds_mm[:,0] 
        and np.all(design <= AntennaBounds_mm[:,1])
        and design[0] /design[1] >= 1.2
    )






































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