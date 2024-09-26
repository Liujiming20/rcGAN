import csv

from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.utils.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS
import numpy as np

from NSGA.evaluate_objective import evaluate_objective
from NSGA.pop_ind_class import Individual, Population
from utils.data_process import VectorProcessor, LabelProcessor, ProfileProcessor


def generate_initial_samples(pop, sample_path, sample_add_path, random_index):
    bottom_beam_section_type = IntegerVariable(15, 24)
    bottom_column_section_type = IntegerVariable(12, 21)

    damper_position1 = IntegerVariable(-1, 20)
    damper_position2 = IntegerVariable(-1, 20)
    damper_position3 = IntegerVariable(-1, 20)
    damper_position4 = IntegerVariable(-1, 20)
    damper_position5 = IntegerVariable(-1, 20)
    damper_position6 = IntegerVariable(-1, 20)
    damper_position7 = IntegerVariable(-1, 20)
    damper_position8 = IntegerVariable(-1, 20)
    damper_position9 = IntegerVariable(-1, 20)
    damper_position10 = IntegerVariable(-1, 20)
    damper_position11 = IntegerVariable(-1, 20)
    damper_position12 = IntegerVariable(-1, 20)
    damper_position13 = IntegerVariable(-1, 20)
    damper_position14 = IntegerVariable(-1, 20)
    damper_position15 = IntegerVariable(-1, 20)
    damper_position16 = IntegerVariable(-1, 20)
    damper_position17 = IntegerVariable(-1, 20)
    damper_position18 = IntegerVariable(-1, 20)
    damper_position19 = IntegerVariable(-1, 20)
    damper_position20 = IntegerVariable(-1, 20)
    damper_position21 = IntegerVariable(-1, 20)

    damper_coefficient = FloatVariable(630.0, 1890.0)
    damper_stiffness = FloatVariable(625.0, 1875.0)

    design_space = DesignSpace([bottom_beam_section_type, bottom_column_section_type, damper_position1, damper_position2, damper_position3,
         damper_position4, damper_position5, damper_position6, damper_position7, damper_position8, damper_position9,
         damper_position10, damper_position11, damper_position12, damper_position13, damper_position14,
         damper_position15, damper_position16, damper_position17, damper_position18, damper_position19,
         damper_position20, damper_position21, damper_coefficient, damper_stiffness])

    sampling = MixedIntegerSamplingMethod(LHS, design_space, criterion="ese", random_state=random_index)
    n_doe = pop
    Xt = sampling(n_doe)

    np.savetxt(sample_path, design_space.decode_values(Xt), delimiter=",")

    n_add = int(pop/3)
    x_new = sampling.expand_lhs(Xt, n_add, method="ese")
    np.savetxt(sample_add_path, design_space.decode_values(x_new), delimiter=",")


def modify_initial_samples(sample_filepath, sample_del_path):
    input_data = np.loadtxt(sample_filepath, delimiter=",", dtype=float)

    out_data = []
    for row in range(len(input_data)):
        beam_profile_index = int(input_data[row][0])
        column_profile_index = int(input_data[row][1])

        if beam_profile_index - column_profile_index > 6:
            continue
        else:
            out_data.append(input_data[row])

    with open(sample_del_path, "w+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(out_data)


def get_pop_normal(pop, initial_sample):
    initial_pop1 = np.empty([pop, 29], dtype=float)
    for i in range(pop):
        input_array = initial_sample[i]
        beam_pro_index = input_array[0]
        column_pro_index = input_array[1]

        input_array1 = np.insert(input_array, 0, beam_pro_index - 1)
        input_array2 = np.insert(input_array1, 0, beam_pro_index - 2)

        input_array3 = np.insert(input_array2, 3, column_pro_index - 1)
        input_array4 = np.insert(input_array3, 3, column_pro_index - 2)

        initial_pop1[i] = input_array4

    initial_pop_x1 = initial_pop1[:, 0:6]
    initial_pop_x2 = initial_pop1[:, 6:27]
    initial_pop_x3 = initial_pop1[:, 27:]

    profile_processor = ProfileProcessor(initial_pop_x1)
    vector_processor = VectorProcessor(initial_pop_x2)
    pop_new = np.hstack((profile_processor.x_1_conversion, vector_processor.x_2_vectors, initial_pop_x3))
    label_processor = LabelProcessor()
    initial_pop_normal = label_processor.pre_process(pop_new)

    return initial_pop1, initial_pop_normal


def initialize_pop(pop, initial_sample, value_cal):
    population = Population()

    for i in range(pop):
        individual = Individual()
        individual.features = initial_sample[i]
        individual.objectives = value_cal[i]

        population.append(individual)

    return population


def initialize_variables(pop, random_index):
    sample_path = './NSGA/source_data/NSGA_2_samples/sample_NSGA.csv'
    sample_add_path = './NSGA/source_data/NSGA_2_samples/sample_NSGA_add.csv'
    modify_sample_path = "./NSGA/source_data/NSGA_2_samples/sample_NSGA_del.csv"
    modify_sample_add_path = "./NSGA/source_data/NSGA_2_samples/sample_NSGA_add_del.csv"

    net_para_reg = [4, 256,128,64,32,16]
    reg_pth_path = "./NSGA/source_data/networks/reg/pre-train_reg.pth"
    net_para_gen = [4, 256,128,64,32,16]
    gen_pth_path = "./NSGA/source_data/networks/gen/generator_lf_smoothL1_atten2.pth"

    generate_initial_samples(pop, sample_path, sample_add_path, random_index)

    modify_initial_samples(sample_path, modify_sample_path)
    modify_initial_samples(sample_add_path, modify_sample_add_path)

    sample_ini = np.loadtxt(modify_sample_path, delimiter=",", dtype=np.float32)
    sample_add = np.loadtxt(modify_sample_add_path, delimiter=",", dtype=np.float32)

    add_num = pop - len(sample_ini)

    if add_num > len(sample_add):
        raise SystemExit('The qualified individuals in the supplementary sample are not enough to fill the missing initial sample,\n'
                         ' please increase the number of supplementary samples.')

    sample_add = sample_add[0:add_num, :]
    initial_sample = np.concatenate((sample_ini, sample_add), axis=0)

    ini_pop_data, pop_normal_data = get_pop_normal(pop, initial_sample)

    value_cal = evaluate_objective(ini_pop_data, pop_normal_data, net_para_reg, reg_pth_path, net_para_gen, gen_pth_path, pop, ini_option=True)

    population = initialize_pop(pop, initial_sample, value_cal)

    return population