from dataclasses import dataclass
import math
import numpy as np
RelativePermittivity: float = 4.4
c: float = 3.0e8
# @dataclass(frozen=True)
class RMA:
    # define the class of antenna and contain all the parameters and formula
 
    def __init__(self, width_mm: float, length_mm:float ,thickness_mm:float):
        self.width_mm = width_mm
        self.length_mm = length_mm
        self.thickness_mm = thickness_mm
        self.volume_mm3 = width_mm*length_mm*thickness_mm
        self.RelativePermittivityEff = (RelativePermittivity+1)/2 + (RelativePermittivity-1)*(((1+12 * thickness_mm/width_mm))**(-1/2))/2
        self.lengthDelta_mm = 0.412*thickness_mm *((self.RelativePermittivityEff+0.3)*(width_mm/thickness_mm+0.264))/((self.RelativePermittivityEff-0.258)*(width_mm/thickness_mm+0.8))
        self.lengthEff_mm = length_mm + 2*self.lengthDelta_mm
        self.ResonanceF = c/(2*(self.lengthEff_mm*1e-3)*math.sqrt(self.RelativePermittivityEff))
        
