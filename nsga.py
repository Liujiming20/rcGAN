import os
import shutil

import numpy
import torch

from NSGA.evolve import evolve_process
from NSGA.initialize_variables import initialize_variables
from NSGA.non_domination_sort_mod import non_domination_sort_mod

from NSGA.plot_front import plot_front_final

import timeit


def main():
    pop = 400
    # pop = 1000
    gen = 500
    # gen = 200
    LTH_random_index = 1

    numpy.random.seed(111)
    torch.manual_seed(111)

    chromosome = initialize_variables(pop, LTH_random_index)

    chromosome = non_domination_sort_mod(chromosome, True)

    file_root = "./NSGA/opt_result/LHS_{}/".format(str(LTH_random_index))

    if os.path.exists(file_root):
        shutil.rmtree(file_root)
    else:
        # 如果不存在，则创建文件夹
        os.makedirs(file_root)

    evol = evolve_process(chromosome, pop, gen, file_root)

    # evol = None

    # opt_result_path = "./NSGA/opt_result/front0.csv"
    # plot_front_final(evol, opt_result_path)


if __name__ == '__main__':
    main()


# # 使用timeit测量main函数的执行时间
# elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=10)
#
# print(f"main函数平均运行时间: {elapsed_time / 10:.6f} 秒")

