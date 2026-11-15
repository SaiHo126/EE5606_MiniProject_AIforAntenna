from operator import index
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
if __package__:
    from .PatchFormula import RMA
else:
    from PatchFormula import RMA

crossoverRate = 0.8
mutationRate = 0.2
populationSize = 6
chomosomes = np.array([])
AntennaBounds_mm = np.array([[25.0, 50.0], [25.0, 45.0], [1.0, 5.0]])

class GeneticAlgorithm:
    def __init__(self, Bounds: np.ndarray, seed: int = 42) -> None:
        self.Bounds = Bounds
        self.objective = RMA.fitness
        # Seed once per GA instance. Every operator advances this same sequence.
        self.rng = np.random.default_rng(seed)

    class config:
        populationSize: int = 3
        generationsMin: int = 20
        tournamentSize: int = 2
        crossoverRate: float = 0.8
        mutationRate: float = 0.2
        stepSize: float = 0.1
        eliteCount: int = 1
        seed: int = 5
        terminationCondition: float = 0.01
        frequencyPenalty: float = 1.0
        def __init__(self, populationSize: int, generationsMin: int, tournamentSize: int, crossoverRate: float, mutationRate: float, stepSize: float, eliteCount: int, seed: int):
            self.populationSize = populationSize
            self.generationsMin = generationsMin
            self.tournamentSize = tournamentSize
            self.crossoverRate = crossoverRate
            self.mutationRate = mutationRate
            self.stepSize = stepSize
            self.eliteCount = eliteCount
            self.seed = seed
    class results:
        bestDesign_mm: RMA
        bestFitness: float
        bestScoreHistory: list[float]
        iterativeNumber: int
        elpasedTime: float
    def initializePopulation(self, config) -> np.ndarray:
        chomosomes = []
        for _ in range(config.populationSize):
            width_mm = self.rng.uniform(self.Bounds[0,0], self.Bounds[0,1])
            length_mm = self.rng.uniform(self.Bounds[1,0], self.Bounds[1,1])
            thickness_mm = self.rng.uniform(self.Bounds[2,0], self.Bounds[2,1])
            antenna = RMA(width_mm, length_mm, thickness_mm)
            antenna.geometryFix()
            # Correct tiny rounding errors at the W/L = 1.2 boundary.
            if antenna.width_mm / antenna.length_mm < 1.2:
                antenna.width_mm = np.nextafter(antenna.width_mm, np.inf)
            # Recalculate frequency and volume using the repaired dimensions.
            antenna = RMA(antenna.width_mm, antenna.length_mm, antenna.thickness_mm)
            chomosomes = np.append(chomosomes, antenna)
        return chomosomes
    def select_parent(self, population: np.ndarray, config, frequencyPlenalty: float = 1.0) -> RMA:
        # Sample distinct competitors within this tournament.
        competitors = self.rng.choice(len(population), size=config.tournamentSize, replace=False)
        candidates = []
        for index in competitors:
            antenna = population[index]
            # Rebuild from current dimensions: geometryFix may have changed them
            # without recalculating the stored volume and resonance frequency.
            candidates.append(RMA(antenna.width_mm, antenna.length_mm, antenna.thickness_mm))
        # self.objective is RMA.fitness; the new object also protects the original
        # population from later changes to the selected parent.
        return min(candidates, key=lambda antenna: self.objective(antenna, frequencyPlenalty))
    def crossover(self, parent1: RMA, parent2: RMA, config:config) -> np.ndarray:
        genes1 = np.array([parent1.width_mm, parent1.length_mm, parent1.thickness_mm])
        genes2 = np.array([parent2.width_mm, parent2.length_mm, parent2.thickness_mm])
        if self.rng.random() < config.crossoverRate:
            alpha = self.rng.random()
            child1_genes = alpha * genes1 + (1 - alpha) * genes2
            child2_genes = alpha * genes2 + (1 - alpha) * genes1
        else:
            child1_genes = genes1
            child2_genes = genes2
        # Blending valid parents can round W/L just below 1.2.
        for genes in (child1_genes, child2_genes):
            if genes[0] / genes[1] < 1.2:
                genes[0] = np.nextafter(genes[0], np.inf)
        return np.array([RMA(*child1_genes), RMA(*child2_genes)])
    def mutate(self, child: RMA, config) -> RMA:
        genes = np.array([child.width_mm, child.length_mm, child.thickness_mm], dtype=float)
        ranges = self.Bounds[:, 1] - self.Bounds[:, 0]
        for i in range(len(genes)):
            if self.rng.random() < config.mutationRate:
                change_mm = self.rng.normal(
                    loc=0.0,
                    scale=config.stepSize * ranges[i]
                )
                genes[i] += change_mm
        # Ensure positive dimensions within the allowed bounds.
        genes = np.clip(genes, self.Bounds[:, 0], self.Bounds[:, 1])
        mutated = RMA(*genes)
        mutated.geometryFix()
        if mutated.width_mm / mutated.length_mm < 1.2:
            mutated.width_mm = np.nextafter(mutated.width_mm, np.inf)
        # Refresh calculated properties after geometry repair.
        return RMA(mutated.width_mm, mutated.length_mm, mutated.thickness_mm)
    def create_next_generation(self,population: np.ndarray, config: config, eliteNumber: int, frequencyPlenalty: float = 1) -> np.ndarray:
        score = np.array([self.objective(antenna, frequencyPlenalty) for antenna in population])
        order = np.argsort(score)
        next_population = []
        seen = set()
        for i in order:
            if len(next_population) >= eliteNumber:
                break
            elite = population[i]
            genes = (elite.width_mm, elite.length_mm, elite.thickness_mm)
            if genes not in seen:
                next_population.append(RMA(*genes))
                seen.add(genes)
        while len(next_population) < config.populationSize:
            parent1 = self.select_parent(population, config, frequencyPlenalty)
            parent2 = self.select_parent(population, config, frequencyPlenalty)
            for child in self.crossover(parent1, parent2, config):
                child = self.mutate(child, config)
                genes = (child.width_mm, child.length_mm, child.thickness_mm)
                if genes in seen:
                    # Replace duplicates even if mutation is off or both parents match.
                    child = RMA(*self.rng.uniform(self.Bounds[:, 0], self.Bounds[:, 1]))
                    child.geometryFix()
                    if child.width_mm / child.length_mm < 1.2:
                        child.width_mm = np.nextafter(child.width_mm, np.inf)
                    genes = (child.width_mm, child.length_mm, child.thickness_mm)
                    child = RMA(*genes)
                if genes not in seen:
                    next_population.append(child)
                    seen.add(genes)
                if len(next_population) == config.populationSize:
                    break
        return np.array(next_population)
    def run(self, config: config) -> results:
        results = self.results()
        # 1.starting time and create initial population
        started = perf_counter()
        population = self.initializePopulation(config)
        # 2.Score each antenna design point, lower score is better
        # save the best design for elite selection and return
        populationFitness = np.array([self.objective(antenna, config.frequencyPenalty) for antenna in population])
        eliteGenes = population[np.argmin(populationFitness)]
        bestFitness = np.min(populationFitness)
        results.bestDesign_mm = RMA(eliteGenes.width_mm, eliteGenes.length_mm, eliteGenes.thickness_mm)
        results.bestScoreHistory = [bestFitness]
        results.iterativeNumber = 0
        # Create next generation
        fitnessChange = np.inf
        while fitnessChange >= config.terminationCondition:
            population =self.create_next_generation(population, config, config.eliteCount, config.frequencyPenalty)
            populationFitness = np.array([self.objective(antenna, config.frequencyPenalty) for antenna in population])
            eliteGenes = population[np.argmin(populationFitness)]
            bestFitness = np.min(populationFitness)
            results.bestDesign_mm = population[np.argmin(populationFitness)]
            results.bestScoreHistory.append(bestFitness)
            results.iterativeNumber += 1
            if len(results.bestScoreHistory) > config.generationsMin:
                fitnessChange = abs(results.bestScoreHistory[-1] - results.bestScoreHistory[-2])/abs(results.bestScoreHistory[-2])
        results.elpasedTime = perf_counter() - started
        results.bestFitness = bestFitness
        return results

        
        


if __name__ == "__main__":
    config = GeneticAlgorithm.config(
        populationSize=populationSize,
        generationsMin=100,
        tournamentSize=2,
        crossoverRate=crossoverRate,
        mutationRate=mutationRate,
        stepSize=0.1,
        eliteCount=1,
        seed=5
    )
    results = GeneticAlgorithm(AntennaBounds_mm).run(config)
    generations = range(len(results.bestScoreHistory))
    plt.plot(generations, results.bestScoreHistory)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm convergence")
    plt.show()
