from dataclasses import dataclass
import math
import numpy as np
RelativePermittivity: float = 4.4
c: float = 3.0e8
AntennaBounds_mm = np.array([[25.0, 50.0], [25.0, 45.0], [1.0, 5.0]])
# @dataclass(frozen=True)
class RMA:
    # define the class of antenna and contain all the parameters and formula
 
    def __init__(self, width_mm: float, length_mm:float ,thickness_mm:float):
        # Initialize the antenna design with given dimensions and calculate derived properties
        self.width_mm = width_mm
        self.length_mm = length_mm
        self.thickness_mm = thickness_mm
        self.volume_mm3 = width_mm*length_mm*thickness_mm
        self.RelativePermittivityEff = (RelativePermittivity+1)/2 + (RelativePermittivity-1)*(((1+12 * thickness_mm/width_mm))**(-1/2))/2
        self.lengthDelta_mm = 0.412*thickness_mm *((self.RelativePermittivityEff+0.3)*(width_mm/thickness_mm+0.264))/((self.RelativePermittivityEff-0.258)*(width_mm/thickness_mm+0.8))
        self.lengthEff_mm = length_mm + 2*self.lengthDelta_mm
        self.ResonanceF_GHz = c/(2*(self.lengthEff_mm*1e-3)*math.sqrt(self.RelativePermittivityEff))/1e9
    def geometryCheck(self) -> bool:
        # Check if the design is within the bounds & W/L >= 1.2
        design = np.array([self.width_mm, self.length_mm, self.thickness_mm])
        if design.shape != (3,) or np.all(np.isfinite(design)) == False:
            print("Design dimension is not correct or contains non-finite values.")
            return False
        return bool( 
            np.all(design >= AntennaBounds_mm[:,0]) 
            and np.all(design <= AntennaBounds_mm[:,1])
            and design[0] /design[1] >= 1.2
        )
    def geometryFix(self) -> None:
        # Fix the design if geometryCheck fails
        maximum_length = np.nextafter(50.0 / 1.2, -np.inf)
        if self.geometryCheck() == False:
            self.length_mm = np.clip(self.length_mm, AntennaBounds_mm[1,0] , maximum_length)
            self.width_mm = np.clip(self.width_mm, 1.2*self.length_mm, AntennaBounds_mm[0,1])
            self.thickness_mm = np.clip(self.thickness_mm, AntennaBounds_mm[2,0], AntennaBounds_mm[2,1])
    def fitness(self, frequencyPlenalty: float) -> float:
        # Define the fitness value of the design, which is a combination of volume and resonance frequency penalty
        return self.volume_mm3/11250 + frequencyPlenalty * abs(self.ResonanceF_GHz - 2.4)
