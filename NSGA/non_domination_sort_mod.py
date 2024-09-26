def cal_crowding_distance(front):
    solutions_num = len(front)  # Number of individuals in this layer

    for individual in front:
        individual.crowding_distance = 0

    for m in range(len(front[0].objectives)):  # three objective function
        front.sort(key=lambda individual: individual.objectives[
            m])  # Sort by the m th target value from smallest to largest.

        # The crowding of the first and last two individuals is infinite.
        # front[0].crowding_distance = 10**9  # Here += seems more appropriate
        # front[solutions_num-1].crowding_distance = 10**9
        front[0].crowding_distance += 10 ** 9
        front[solutions_num - 1].crowding_distance += 10 ** 9

        m_values = [individual.objectives[m] for individual in
                    front]  # Puts the I-th target value of all individuals in that layer into an array.
        scale = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        # Calculate the degree of crowding between an individual in the middle of the layer and other individuals.
        for i in range(1, solutions_num - 1):
            front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale


def non_domination_sort_mod(population_input, option):
    chromosome = population_input
    chromosome.fronts = [[]]

    for individual in chromosome.population:
        individual.domination_count = 0
        individual.dominated_solutions = []

        for other_individual in chromosome.population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            chromosome.fronts[0].append(individual)  # Get individuals on the first frontier

    # By using the elimination method, the individual controlled by the count layer is excluded.
    # If the number controlled by the count layer individual is 0 after the count layer individual is eliminated,
    # it means that it is an individual of the count+1 layer.
    count = 0
    while len(chromosome.fronts[count]) > 0:
        temp = []
        for individual in chromosome.fronts[count]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1

                if other_individual.domination_count == 0: # This means that these individuals are the dominant individuals at the count+1 level.
                    other_individual.rank = count + 1
                    temp.append(other_individual)

        count += 1
        if len(temp) > 0:
            chromosome.fronts.append(temp)
        else:
            break

    if option:
        # Order the particles according to how crowded they are.
        for front in chromosome.fronts:
            cal_crowding_distance(front)

    return chromosome



