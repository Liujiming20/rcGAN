import random

import numpy
import numpy as np
from tqdm import tqdm

from NSGA.evaluate_objective import evaluate_objective
from NSGA.initialize_variables import get_pop_normal
from NSGA.non_domination_sort_mod import non_domination_sort_mod, cal_crowding_distance
from NSGA.plot_front import plot_front_process
from NSGA.pop_ind_class import Individual, Population


def crowding_operator(individual, other_individual):
    # 如果A个体帕累托排名小于B个体，或者A个体与B个体同处一个帕累托面但A的拥挤距离大于B
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1


def choose_with_prob(tournament_prob):
    if random.random() <= tournament_prob:
        return True
    return False


def tournament(chromosome, tournament_prob):
    participants = random.sample(chromosome.population, 2)

    best = None
    for participant in participants:
        # 如果best为空，或best不为空但帕累托排序和拥挤度排序不如B且随即验证通过，则best取当前个体
        if best is None or (crowding_operator(participant, best) == 1 and choose_with_prob(tournament_prob)):
            best = participant

    return best


def get_beta(crossover_param):
    u = random.random()
    if u <= 0.5:
        return (2 * u) ** (1 / (crossover_param + 1))
    return (2 * (1 - u)) ** (-1 / (crossover_param + 1))


def crossover(individual1, individual2, crossover_param):
    child1 = Individual()
    child2 = Individual()

    beta_array = np.empty(25)
    for i in range(25):
        beta_array[i] = get_beta(crossover_param)

    x1_array = np.multiply(np.add(individual1.features, individual2.features), 0.5)
    x2_array = np.multiply(np.subtract(individual1.features, individual2.features), 0.5)

    child1.features = np.add(x1_array, np.multiply(x2_array, beta_array))
    child2.features = np.subtract(x1_array, np.multiply(x2_array, beta_array))

    return child1, child2


def get_delta(mutation_param):
    u = random.random()
    if u < 0.5:
        return u, (2 * u) ** (1 / (mutation_param + 1)) - 1
    return u, 1 - (2 * (1 - u)) ** (1 / (mutation_param + 1))


def mutate(child, mutation_param, variables_range):
    num_of_features = len(child.features)
    for gene in range(num_of_features):
        u, delta = get_delta(mutation_param)
        if u < 0.5:
            child.features[gene] += delta * (child.features[gene] - variables_range[gene][0])  # 可行域下界
        else:
            child.features[gene] += delta * (variables_range[gene][1] - child.features[gene])  # 可行域上界

        if gene <= 22:
            child.features[gene] = int(child.features[gene])

        if child.features[gene] < variables_range[gene][0]:
            child.features[gene] = variables_range[gene][0]
        elif child.features[gene] > variables_range[gene][1]:
            child.features[gene] = variables_range[gene][1]


def create_children(chromosome, tournament_prob, crossover_param, mutation_param, variables_range):
    children = []

    children_sample_input = np.array([])
    option = True
    while len(children) < len(chromosome):  # private method __len__ of the Population
        parent1 = tournament(chromosome, tournament_prob)  # 2个体锦标精英赛选择

        parent2 = parent1  # 初始化父辈2，以便让父辈2可以和父辈1比较；这个循环是为了确保父辈2一定与父辈1不相同
        while parent1 == parent2:
            parent2 = tournament(chromosome, tournament_prob)

        child1, child2 = crossover(parent1, parent2, crossover_param)

        mutate(child1, mutation_param, variables_range)
        mutate(child2, mutation_param, variables_range)

        children.append(child1)
        children.append(child2)

        if option:
            children_sample_input = child1.features.reshape(1, -1)
            option = False
        else:
            children_sample_input = np.append(children_sample_input, child1.features.reshape(1, -1), axis=0)

        children_sample_input = np.append(children_sample_input, child2.features.reshape(1, -1), axis=0)

    return children, children_sample_input


def evolve_process(chromosome, pop, gen, file_root):
    tournament_prob = 0.9
    crossover_param = 2
    mutation_param = 5

    variables_range = [[15, 24],[12, 21],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20],[-1, 20], [630.0, 1890.0],[625.0, 1875.0]]

    net_para_reg = [4, 256,128,64,32,16]
    reg_pth_path = "./NSGA/source_data/networks/reg/pre-train_reg.pth"
    net_para_gen = [4, 256,128,64,32,16]
    gen_pth_path = "./NSGA/source_data/networks/gen/generator_lf_smoothL1_atten2.pth"

    file_path_list = [file_root+"gen0_front0", file_root+"gen100_front0",
                      file_root+"gen200_front0", file_root+"gen300_front0",
                      file_root+"gen400_front0", file_root+"gen500_front0"]

    color_list = ["black", "grey", "tan", "orange", "lightcoral", "red"]

    childern, children_sample_input = create_children(chromosome, tournament_prob, crossover_param, mutation_param, variables_range)

    ini_pop_data, pop_normal_data = get_pop_normal(pop, children_sample_input)
    value_cal = evaluate_objective(ini_pop_data, pop_normal_data, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option=False)

    for i in range(len(childern)):
        childern[i].objectives = value_cal[i]

    returned_population = None
    count = 0
    for i in tqdm(range(gen)):
        chromosome.extend(childern)

        chromosome = non_domination_sort_mod(chromosome, False)  # The number of individuals in this population is 2N.

        new_chromosome = Population()
        front_count = 0
        while len(new_chromosome) + len(chromosome.fronts[front_count]) <= pop:
            cal_crowding_distance(chromosome.fronts[front_count])

            new_chromosome.extend(chromosome.fronts[front_count])
            front_count += 1

        cal_crowding_distance(chromosome.fronts[front_count])
        chromosome.fronts[front_count].sort(key=lambda individual : individual.crowding_distance, reverse=True)

        new_chromosome.extend(chromosome.fronts[front_count][0 : (pop - len(new_chromosome))])

        returned_population = chromosome
        chromosome = new_chromosome  # The number of individuals in this population is N.
        chromosome = non_domination_sort_mod(chromosome, True)

        childern, children_sample_input_new = create_children(chromosome, tournament_prob, crossover_param, mutation_param, variables_range)
        ini_pop_data_new, pop_normal_data_new = get_pop_normal(pop, children_sample_input_new)
        value_cal = evaluate_objective(ini_pop_data_new, pop_normal_data_new, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option=False)
        for j in range(len(childern)):
            childern[j].objectives = value_cal[j]

        if i == 0:
            evol = returned_population.fronts[0]
            plot_front_process(evol, file_path_list, color_list, count)
            count += 1
        # elif (i+1) % 40 == 0:
        elif (i + 1) % 100 == 0:
            evol = returned_population.fronts[0]
            plot_front_process(evol, file_path_list, color_list, count)
            count += 1
        else:
            pass
