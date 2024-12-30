import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import *


def filter(start_moves, max_dur=0):
# filters by end position
    n_moves = start_moves.shape[0]
    xlim = [-3, 3]
    ylim = [-3, 3]
    idx_del = []
    for i0 in range(n_moves):
        end_i0 = np.sum(start_moves[i0], axis=0, keepdims=True)
        if (end_i0 == [0, 0]).all():
            idx_del.append(i0)
            continue

        if not (xlim[0] <= end_i0[0, 0] <= xlim[1] and ylim[0] <= end_i0[0, 1] <= ylim[1]):
            idx_del.append(i0)
            continue

    idx_del = np.unique(np.array(idx_del))
    idx_keep = np.setdiff1d(np.arange(n_moves), idx_del)
    start_moves = start_moves[idx_keep]

    return start_moves


#
max_dur = 6

start_x = np.array([-3, -2, -1, 0, 1, 2, 3])
start_y = np.array([-3, -2, -1, 0, 1, 2, 3])
distances = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
x_act = np.stack((distances, np.zeros(distances.shape[0])), axis=1)
y_act = np.stack((np.zeros(distances.shape[0]), distances), axis=1)
acts_all = np.concatenate((x_act, y_act), axis=0)
n_acts = acts_all.shape[0]
np.random.seed(0)

start_x_, start_y_ = np.meshgrid(start_x, start_y)
starts_all = np.array(list(zip(start_x_.flatten(), start_y_.flatten())))
ends_all = starts_all

start_moves = []
for i0 in range(len(starts_all)):
    for i1 in range(acts_all.shape[0]):
        start_moves.append([starts_all[i0, :], acts_all[i1, :]])

start_moves = np.array(start_moves)


start_moves = filter(start_moves, max_dur=max_dur)
start_move_ends = np.zeros((start_moves.shape[0], 2))
for i0 in range(start_move_ends.shape[0]):
    start_move_ends[i0, :] = np.sum(start_moves[i0], axis=0)



### plot
plot = 0
if plot == True:
    ends = [start_move_ends]
    fig, ax = plt.subplots(1, len(ends), figsize=(20, 8))
    distribution_all = np.zeros((len(ends), 11, 11))

    for i0 in range(len(ends)):
        ends_i0 = ends[i0]
        for i1 in range(ends_i0.shape[0]):
            tmp = ends_i0[i1, :]
            distribution_all[i0, int(tmp[0] + 5), int(tmp[1] + 5)] += 1
        distribution_all[i0, :, :] = np.flip(distribution_all[i0, :, :].T, 0)

        distribution_i0 = np.sqrt(distribution_all[i0, :, :] / np.max(distribution_all[i0, :, :].flatten()))
        for row in range(11):
            for col in range(11):
                radius = distribution_i0[row, col]
                ax.add_patch(
                    Circle((col, 10 - row), radius / 2, edgecolor=(0, 0, 0, 0), facecolor=(0, 0, 0, 1), linewidth=1))

        ax.set_title('Total = ' + str(len(ends_i0)))
        ax.set_xticks(np.arange(0, 11), labels=[])
        ax.set_yticks(np.arange(0, 11), labels=[])
        ax.grid()
    plt.show()

seed = 0


training_set2 = []

two_step_moves_list = []
for i0 in range(start_moves.shape[0]):
    two_step_moves_list.append(start_moves[i0].tolist())

idx_pick_list = []
for x in range(-3, 4):
    for y in range(-3, 4):
        idxs_pick = np.where(np.sum(start_move_ends == [x, y], axis=1) == 2)[0]
        if idxs_pick.shape[0] != 0:
            idx_pick = int(np.random.choice(idxs_pick, 1))

            training_set2.append(two_step_moves_list[idx_pick])
            idx_pick_list.append(idx_pick)
        else:
            a = 1


moves = [training_set2]
trained_targets = []
for i0 in range(len(moves)):
    for i1 in range(len(moves[i0])):
        trained_targets.append(moves[i0][i1])
        pass


p1 = []
for i0 in range(len(trained_targets)):
    p1.append(1)

trained_target_probs = []
for i0 in range(len(trained_targets)):
    trained_target_probs.append(0)
trained_target_probs.append(1)


list_len = []
for i0 in range(len(trained_targets)):
    list_len.append(np.sum(abs(np.array(trained_targets[i0][1]))))
list_len = np.array(list_len)
alpha_group_id_list = np.zeros(len(trained_targets), dtype=int)

for i0 in range(len(trained_targets)):
    alpha_group_id_list[i0] = int(list_len[i0]-1)
alpha_group_id_list = list(alpha_group_id_list)
alpha_list = list(np.ones(len(np.unique(np.array(alpha_group_id_list))), dtype=int))






print(trained_targets)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)


###
validation_targets = [[[-3, -3], [0, 6]], [[-2, -2], [4, 0]], [[3, -3], [-1, 0]], [[1, 3], [0, -2]]]
rep = 5

max_alpha_id = max(alpha_group_id_list)
for i0 in range(len(validation_targets)):
    trained_targets.append(validation_targets[i0])
    p1.append(rep)
    trained_target_probs.insert(-1, 0)
    alpha_group_id_list.append(max_alpha_id+1)
alpha_list.append(0)

print(trained_targets)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)


all_start = np.meshgrid(x)
pass
