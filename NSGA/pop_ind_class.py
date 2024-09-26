class Population():
    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):

        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)


class Individual(object):
    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None  # The amount controlled by other individuals
        self.dominated_solutions = None  # Dominate other individuals
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features.tolist() == other.features.tolist()
        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)