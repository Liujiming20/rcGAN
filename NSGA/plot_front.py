import csv

import numpy as np
from matplotlib import pyplot as plt


def plot_front_final(evol, file_path):
    func = [i.objectives for i in evol]

    with open(file_path, "w+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(func)

    # func = np.loadtxt(file_path, delimiter=",", dtype=float)

    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    function3 = [i[2] for i in func]

    # function1 = [1,2,3,4,5,6,7,8,9]
    # function2 = [4,5,6,1,2,3,4,5,6]
    # function3 = [7,8,9,2,3,4,5,6,7]

    fig = plt.figure()
    ax3d = fig.add_subplot(projection='3d')
    # ax3d.set_yscale("log")
    # ax3d.set_zscale("log")
    ax3d.set_xlim(0.01, 0.015)
    ax3d.set_ylim(4500, 6000)
    ax3d.set_zlim(500000, 2300000)

    # ax3d.scatter(function1, function2, function3)
    # ax3d.scatter(function1, function2, 0, zdir='z', c='r')
    # ax3d.scatter(function1, function3, 0, zdir='y', c='g')
    # ax3d.scatter(function2, function3, 0, zdir='x', c='b')

    ax3d.ticklabel_format(style='sci', scilimits=(-1, 2), axis="z", useMathText=True)
    ax3d.scatter(function1, function2, function3)
    ax3d.plot(function1, function2, 500000, zdir='z', c='r')
    ax3d.plot(function1, function3, 6000, zdir='y', c='g')
    ax3d.plot(function2, function3, 0.01, zdir='x', c='b')

    plt.show()


def plot_front_process(evol, file_path_list, color_list, count):
    func = [i.objectives for i in evol]
    fea = [i.features for i in evol]

    object_filepath = file_path_list[count] + "_objective.csv"
    feature_filepath = file_path_list[count] + "_feature.csv"

    with open(object_filepath, "w+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(func)

    with open(feature_filepath, "w+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(fea)

    # count_local = count + 1
    #
    # if count_local == len(file_path_list):
    #     f_1 = []
    #     f_2 = []
    #     f_3 = []
    #
    #     for i in range(len(file_path_list)):
    #         func = np.loadtxt((file_path_list[i] + "_objective.csv"), delimiter=",", dtype=float)
    #
    #         function1 = [i[0] for i in func]
    #         function2 = [i[1] for i in func]
    #         function3 = [i[2] for i in func]
    #
    #         f_1.append(function1)
    #         f_2.append(function2)
    #         f_3.append(function3)
    #
    #     fig = plt.figure()
    #     ax3d = fig.add_subplot(projection='3d')
    #     # ax3d.set_yscale("log")
    #     # ax3d.set_zscale("log")
    #     ax3d.set_xlim(0.01, 0.015)
    #     ax3d.set_ylim(4500, 6000)
    #     ax3d.set_zlim(500000, 2300000)
    #
    #     ax3d.ticklabel_format(style='sci', scilimits=(-1, 2), axis="z", useMathText=True)
    #     for i in range(len(color_list)):
    #         function1 = f_1[i]
    #         function2 = f_2[i]
    #         function3 = f_3[i]
    #         ax3d.scatter(function1, function2, function3, c=color_list[i])
    #         ax3d.plot(function1, function2, 500000, zdir='z', c=color_list[i])
    #         ax3d.plot(function1, function3, 6000, zdir='y', c=color_list[i])
    #         ax3d.plot(function2, function3, 0.01, zdir='x', c=color_list[i])
    #         plt.ion()
    #         plt.show()
    #         plt.pause(20)
    #         plt.cla()

        # plt.show()