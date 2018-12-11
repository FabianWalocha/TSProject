import random
import time

def GA_approach(graph, PopulationIntended=100, GenerationIntended=500, mutationRateIntended=0.5, tournamentSelectionSizeIntended=20,
                elitismIntended=True, timed=False):
    mutationRate = mutationRateIntended  # 2
    tournamentSelectionSize = tournamentSelectionSizeIntended

    elitism = elitismIntended

    PopulationSize = PopulationIntended
    GenerationSize = GenerationIntended

    Cities = []

    Matrix = graph.weighted_adjacency_matrix

    class City:
        def __init__(self, id=0):
            self.id = id

        def distanceTo(self, city):
            return Matrix[self.id][city.id]

        def __repr__(self):
            return str("( id = " + str(self.id) + ")")

    class Route:
        def __init__(self, route=None):
            self.route = []
            self.fitness = 0.0
            self.distance = 0
            if route is not None:
                self.route = route
            else:
                for i in range(0, len(Cities)):
                    self.route.append(None)

        def __len__(self):
            return len(self.route)

        def __getitem__(self, index):
            return self.route[index]

        def __setitem__(self, key, value):
            self.route[key] = value

        def __repr__(self):
            geneString = ""
            for i in range(0, len(self.route)):
                geneString += str(self[i]) + "\n "
            return geneString

        def generateIndividual(self):
            for cityIndex in range(0, len(Cities)):
                self.addCity(cityIndex, Cities[cityIndex])
            random.shuffle(self.route)

        def addCity(self, id, city):
            self.route[id] = city
            self.fitness = 0.0
            self.distance = 0

        def getFitness(self):
            if self.fitness == 0:
                self.fitness = 1 / float(self.getDistance())
            return self.fitness

        def getDistance(self):
            if self.distance == 0:
                distance = 0
                len_route = len(self.route)
                B_final = self.route[len_route - 1]

                for i in range(0, len_route - 1):
                    A = self.route[i]
                    B = self.route[i + 1]
                    distance += A.distanceTo(B)

                distance += B_final.distanceTo(self.route[0])  # From Last point(B) To First One
                self.distance = distance
            return self.distance

    class Population:
        def __init__(self, populationSize, initialise):
            self.routes = []
            for i in range(0, populationSize):
                self.routes.append(None)

            if initialise:
                for i in range(0, populationSize):
                    newTour = Route()
                    newTour.generateIndividual()
                    self.routes[i] = newTour

        def __setitem__(self, key, value):
            self.routes[key] = value

        def __getitem__(self, index):
            return self.routes[index]

        def __len__(self):
            return len(self.routes)

        def getFittest(self):
            fittest = self.routes[0]
            for i in range(0, self.populationSize()):
                if fittest.getFitness() <= self[i].getFitness():
                    fittest = self[i]
            return fittest

        def populationSize(self):
            return len(self.routes)

        def getRouletteList(self):  # here in the end of list - sum of all fitnesses (max value)
            roiletlist = []
            a = 0.0
            roiletlist.append(a)
            for i in range(0, self.populationSize()):
                a += self[i].getFitness()
                roiletlist.append(a)

            return roiletlist

    def evolvePopulation(population):
        nextPopulation = Population(len(population), False)
        elitismOffset = 0
        if elitism:
            nextPopulation[0] = population.getFittest()
            elitismOffset = 1

        mattingPopulation = mattingCreatorPool(population)
        # Current work
        for i in range(elitismOffset, nextPopulation.populationSize()):
            parent1 = rouletteSelection(mattingPopulation)
            parent2 = rouletteSelection(mattingPopulation)
            child = crossover2(parent1, parent2)
            nextPopulation[i] = child
        nextPopulation = mutatePopulation(nextPopulation,
                                          elitismOffset)  # If ElitismOffset is 0 we will mutate all population
        return nextPopulation

    def crossover2(parent_1, parent_2):

        list_1 = []

        geneA = int(random.random() * len(parent_1))
        geneB = int(random.random() * len(parent_1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            list_1.append(parent_1[i])

        list_2 = [item for item in parent_2 if item not in list_1]
        list = []
        i_1 = 0
        i_2 = 0

        for i in range(len(Cities)):
            if (i in range(startGene, endGene)):
                list.append(list_1[i_1])
                i_1 += 1
            else:
                list.append(list_2[i_2])
                i_2 += 1

        return Route(list)

    def mutatePopulation(population, offset):  # New Version
        for i in range(offset, population.populationSize()):
            mutateRoute(population[i])
        return population

    def mutateRoute(route):
        if random.random() < mutationRate:
            id_1 = random.randint(0, len(route) - 1)
            id_2 = random.randint(0, len(route) - 1)
            # SWAP
            tmpRoute = route[id_1]
            route.addCity(id_1, route[id_2])
            route.addCity(id_2, tmpRoute)

    def tournamentSelection(population):
        tPopulation = Population(tournamentSelectionSize, False)
        for i in range(0, tournamentSelectionSize):
            id = random.randint(0, len(population) - 1)
            tPopulation[i] = population[id]
        return tPopulation.getFittest()

    def rouletteSelection(population):
        rouletteList = population.getRouletteList()
        maxvalue = rouletteList[population.populationSize() - 1]
        random_id = random.uniform(0.0, maxvalue)
        for i in range(0, len(rouletteList) - 1):
            if (rouletteList[i] <= random_id < rouletteList[i + 1]):
                return population[i]
        return population[len(rouletteList) - 1]

    def mattingCreatorPool(population):
        size = int(len(population) / 2)
        mattingPopulation = Population(size, False)
        for i in range(0, len(mattingPopulation)):
            mattingPopulation[i] = tournamentSelection(population)
        return mattingPopulation

    for id in range(0, len(Matrix)):
        city = City(id)
        Cities.append(city)

    t1 = time.time()
    # Initialize population
    pop = Population(PopulationSize, True)
    initial_distance = pop.getFittest().getDistance()

    # Evolve population for N generations
    pop = evolvePopulation(pop)

    for i in range(0, GenerationSize):
        # YOU CAN CHANGE THE MAXIMUM TIME HERE
        if timed:
            if time.time() - t1 > 600:
                return initial_distance, pop.getFittest().getDistance(), pop.getFittest(), time.time() - t1
        pop = evolvePopulation(pop)

    t2 = time.time()
    final_time = t2 - t1
    final_distance = pop.getFittest().getDistance()
    final_path = pop.getFittest()

    return initial_distance, final_distance, final_path, final_time