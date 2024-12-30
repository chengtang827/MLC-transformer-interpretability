import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import *


def filter(moves, max_dur=0):
    n_steps = moves[0].shape[0]

    # all intermediate and end point is within range, and ending point not zero
    n_moves = moves.shape[0]
    xlim = [-4, 4]
    ylim = [-4, 4]
    idx_del = []
    for i0 in range(n_moves):
        end_i0 = np.sum(moves[i0], axis=0, keepdims=True)
        if (end_i0 == [0, 0]).all():
            idx_del.append(i0)
            continue

        for i1 in range(n_steps):
            end_i1 = np.sum(moves[i0][:i1 + 1, :], axis=0, keepdims=True)
            if not (xlim[0] <= end_i1[0, 0] <= xlim[1] and ylim[0] <= end_i1[0, 1] <= ylim[1]):
                idx_del.append(i0)
                break

    idx_del = np.unique(np.array(idx_del))
    idx_keep = np.setdiff1d(np.arange(n_moves), idx_del)
    moves = moves[idx_keep]

    # no consecutive movements
    n_moves = moves.shape[0]
    idx_del = []
    for i0 in range(n_moves):
        move_i0 = moves[i0]
        dir_pre = False
        for i1 in range(1, n_steps):
            if dir_pre is False:
                dir_pre = np.nan_to_num(move_i0[0, :] / abs(move_i0[0, :]))
            else:
                dir_pre = dir_cur
            dir_cur = np.nan_to_num(move_i0[i1, :] / abs(move_i0[i1, :]))
            if (dir_pre == dir_cur).all():
                idx_del.append(i0)

    idx_del = np.unique(np.array(idx_del))
    idx_keep = np.setdiff1d(np.arange(n_moves), idx_del)
    moves = moves[idx_keep]

    # total length < 8
    n_moves = moves.shape[0]
    idx_del = []
    for i0 in range(n_moves):
        move_i0 = moves[i0]
        duration = np.sum(abs(move_i0))
        if duration > max_dur:
            idx_del.append(i0)

    idx_del = np.unique(np.array(idx_del))
    idx_keep = np.setdiff1d(np.arange(n_moves), idx_del)
    moves = moves[idx_keep]
    return moves


#
max_dur = 8

distances = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
x_act = np.stack((distances, np.zeros(distances.shape[0])), axis=1)
y_act = np.stack((np.zeros(distances.shape[0]), distances), axis=1)
acts = np.concatenate((x_act, y_act), axis=0)
n_acts = acts.shape[0]
np.random.seed(0)
###
one_step_moves = np.ndarray(0, dtype=object)
for i0 in range(n_acts):
    tmp = acts[i0, np.newaxis, :]
    one_step_moves = np.append(one_step_moves, np.ndarray(1, dtype=object), axis=0)
    one_step_moves[-1] = tmp

one_step_moves = filter(one_step_moves, max_dur=max_dur)
one_step_ends = np.zeros((one_step_moves.shape[0], 2))
for i0 in range(one_step_ends.shape[0]):
    one_step_ends[i0, :] = np.sum(one_step_moves[i0], axis=0)

###
two_step_moves = np.ndarray(0, dtype=object)
for i0 in range(n_acts):
    for i1 in range(n_acts):
        tmp = np.array([acts[i0, :], acts[i1, :]])
        two_step_moves = np.append(two_step_moves, np.ndarray(1, dtype=object), axis=0)
        two_step_moves[-1] = tmp

two_step_moves = filter(two_step_moves, max_dur=max_dur)
two_step_ends = np.zeros((two_step_moves.shape[0], 2))
for i0 in range(two_step_ends.shape[0]):
    two_step_ends[i0, :] = np.sum(two_step_moves[i0], axis=0)

###
three_step_moves = np.ndarray(0, dtype=object)
for i0 in range(n_acts):
    for i1 in range(n_acts):
        for i2 in range(n_acts):
            tmp = np.array([acts[i0, :], acts[i1, :], acts[i2, :]])
            three_step_moves = np.append(three_step_moves, np.ndarray(1, dtype=object), axis=0)
            three_step_moves[-1] = tmp

three_step_moves = filter(three_step_moves, max_dur=max_dur)
three_step_ends = np.zeros((three_step_moves.shape[0], 2))
for i0 in range(three_step_ends.shape[0]):
    three_step_ends[i0, :] = np.sum(three_step_moves[i0], axis=0)

### plot
plot = 0
if plot == True:
    ends = [one_step_ends, two_step_ends, three_step_ends]
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
                ax[i0].add_patch(
                    Circle((col, 10 - row), radius / 2, edgecolor=(0, 0, 0, 0), facecolor=(0, 0, 0, 1), linewidth=1))

        ax[i0].set_title('Total = ' + str(len(ends_i0)))
        ax[i0].set_xticks(np.arange(0, 11), labels=[])
        ax[i0].set_yticks(np.arange(0, 11), labels=[])
        ax[i0].grid()
    plt.show()

seed = 0
training_set1 = []
for i0 in range(len(one_step_moves)):
    training_set1.append(one_step_moves[i0].tolist())

training_set2 = []

two_step_trained = np.array(
    [[-1, -3, 1], [-3, -2, 1], [-4, 3, 1], [-4, -4, 1], [-3, 1, 1], [-2, -1, 1], [-2, 4, 1], [-1, 2, 1], [-1, 4, 1],
     [-1, -1, 1], [-1, 2, 0], [-2, 3, 0], [-2, 4, 0], [-4, 3, 0], [-4, -4, 0],
     [-1, -1, 0], [-1, 4, 0], [-3, 1, 0], [-1, -2, 1], [-1, -4, 0], [-3, -3, 1], [-4, 2, 0], [-1, 1, 1], [-4, -1, 0],
     [-2, 1, 0], [-1, 3, 1], [-2, -3, 1], [-3, -4, 0], [-4, 1, 0], [-3, -1, 1], [-3, 2, 1], [-2, 2, 0], [-4, 4, 1],
     [-2, -2, 0], [-1, -2, 0], [-2, -4, 1], [-3, 4, 0], [-4, -2, 0], [-4, -3, 1], [-3, 3, 0], [-2, 1, 1], [-4, 2, 1],
     [-1, 3, 0], [-2, 3, 1], [-2, -1, 0], [-1, -4, 1], [-1, -3, 0], [-2, 2, 1], [-4, 1, 1], [-1, 1, 0], [-3, -4, 1],
     [-2, -3, 0], [-3, -1, 0], [-3, 3, 1], [-3, 2, 0], [-4, -1, 1], [-3, -3, 0], [1, 1, 1],
     [1, 3, 1], [1, 3, 0], [3, 1, 1], [3, 1, 0]])
two_step_tested = [[1, 2], [2, 1], [2, 2], [4, 1]]

two_step_moves_list = []
for i0 in range(two_step_moves.shape[0]):
    two_step_moves_list.append(two_step_moves[i0].tolist())

for x in range(-4, 5):
    for y in range(-4, 5):
        # if this point is in the tested set, ignore
        if [x, y] in two_step_tested:
            continue
        # If this point in the trained set, pick the trained sample
        idx_trained = np.where(np.sum(two_step_trained[:, :2] == [x, y], axis=1) == 2)[0]
        if idx_trained.shape[0] != 0:
            tmp = two_step_trained[np.random.choice(idx_trained, 1), :]
            if tmp[0, 2] == 1:
                training_set2.append([[tmp[0, 0], 0], [0, tmp[0, 1]]])
            else:
                training_set2.append([[0, tmp[0, 1]], [tmp[0, 0], 0]])

        else:  # not in the trained set, randomly pick from repo
            idx_pick = np.where(np.sum(two_step_ends == [x, y], axis=1) == 2)[0]
            if idx_pick.shape[0] != 0:
                training_set2.append(two_step_moves_list[int(np.random.choice(idx_pick, 1))])

training_set3 = []
three_step_moves_list = []
for i0 in range(three_step_moves.shape[0]):
    three_step_moves_list.append(three_step_moves[i0].tolist())

for x in range(-4, 5):
    for y in range(-4, 5):
        idx_pick = np.where(np.sum(three_step_ends == [x, y], axis=1) == 2)[0]
        if idx_pick.shape[0] != 0:
            # np.random.seed(1)
            tmp = three_step_moves_list[int(np.random.choice(idx_pick, 1))]
            training_set3.append(tmp)

moves = [training_set1, training_set2, training_set3]
trained_targets = []
for i0 in range(len(moves)):
    for i1 in range(len(moves[i0])):
        trained_targets.append(moves[i0][i1])
        pass
print(trained_targets)
trained_targets = [[[-4.0, 0.0]], [[-3.0, 0.0]], [[-2.0, 0.0]], [[-1.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]],
                   [[4.0, 0.0]], [[0.0, -4.0]], [[0.0, -3.0]], [[0.0, -2.0]], [[0.0, -1.0]], [[0.0, 1.0]], [[0.0, 2.0]],
                   [[0.0, 3.0]], [[0.0, 4.0]], [[-4, 0], [0, -4]], [[-4, 0], [0, -3]], [[0, -2], [-4, 0]],
                   [[-4, 0], [0, -1]], [[2.0, 0.0], [-6.0, 0.0]], [[0, 1], [-4, 0]], [[-4, 0], [0, 2]],
                   [[0, 3], [-4, 0]], [[-4, 0], [0, 4]], [[-3, 0], [0, -4]], [[0, -3], [-3, 0]], [[-3, 0], [0, -2]],
                   [[0, -1], [-3, 0]], [[1.0, 0.0], [-4.0, 0.0]], [[-3, 0], [0, 1]], [[-3, 0], [0, 2]],
                   [[-3, 0], [0, 3]], [[0, 4], [-3, 0]], [[-2, 0], [0, -4]], [[-2, 0], [0, -3]], [[0, -2], [-2, 0]],
                   [[-2, 0], [0, -1]], [[-4.0, 0.0], [2.0, 0.0]], [[0, 1], [-2, 0]], [[0, 2], [-2, 0]],
                   [[-2, 0], [0, 3]], [[-2, 0], [0, 4]], [[-1, 0], [0, -4]], [[0, -3], [-1, 0]], [[-1, 0], [0, -2]],
                   [[-1, 0], [0, -1]], [[-3.0, 0.0], [2.0, 0.0]], [[0, 1], [-1, 0]], [[0, 2], [-1, 0]],
                   [[0, 3], [-1, 0]], [[-1, 0], [0, 4]], [[0.0, 2.0], [0.0, -6.0]], [[0.0, -4.0], [0.0, 1.0]],
                   [[0.0, 2.0], [0.0, -4.0]], [[0.0, -4.0], [0.0, 3.0]], [[0.0, 2.0], [0.0, -1.0]],
                   [[0.0, -3.0], [0.0, 5.0]], [[0.0, 4.0], [0.0, -1.0]], [[0.0, -1.0], [0.0, 5.0]],
                   [[1.0, 0.0], [0.0, -4.0]], [[0.0, -3.0], [1.0, 0.0]], [[0.0, -2.0], [1.0, 0.0]],
                   [[0.0, -1.0], [1.0, 0.0]], [[2.0, 0.0], [-1.0, 0.0]], [[1, 0], [0, 1]], [[0, 3], [1, 0]],
                   [[1.0, 0.0], [0.0, 4.0]], [[0.0, -4.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, -3.0]],
                   [[0.0, -2.0], [2.0, 0.0]], [[0.0, -1.0], [2.0, 0.0]], [[-2.0, 0.0], [4.0, 0.0]],
                   [[0.0, 3.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 4.0]], [[0.0, -4.0], [3.0, 0.0]],
                   [[3.0, 0.0], [0.0, -3.0]], [[3.0, 0.0], [0.0, -2.0]], [[0.0, -1.0], [3.0, 0.0]],
                   [[4.0, 0.0], [-1.0, 0.0]], [[0, 1], [3, 0]], [[3.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [3.0, 0.0]],
                   [[3.0, 0.0], [0.0, 4.0]], [[4.0, 0.0], [0.0, -4.0]], [[4.0, 0.0], [0.0, -3.0]],
                   [[4.0, 0.0], [0.0, -2.0]], [[4.0, 0.0], [0.0, -1.0]], [[-1.0, 0.0], [5.0, 0.0]],
                   [[0.0, 2.0], [4.0, 0.0]], [[4.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]],
                   [[0.0, -2.0], [-4.0, 0.0], [0.0, -2.0]], [[-2.0, 0.0], [0.0, -3.0], [-2.0, 0.0]],
                   [[-1.0, 0.0], [0.0, -2.0], [-3.0, 0.0]], [[0.0, -1.0], [1.0, 0.0], [-5.0, 0.0]],
                   [[-1.0, 0.0], [1.0, 0.0], [-4.0, 0.0]], [[0.0, 2.0], [-4.0, 0.0], [0.0, -1.0]],
                   [[0.0, -1.0], [0.0, 3.0], [-4.0, 0.0]], [[-2.0, 0.0], [0.0, 3.0], [-2.0, 0.0]],
                   [[-2.0, 0.0], [0.0, 4.0], [-2.0, 0.0]], [[-1.0, 0.0], [0.0, -4.0], [-2.0, 0.0]],
                   [[0.0, -4.0], [0.0, 1.0], [-3.0, 0.0]], [[-3.0, 0.0], [0.0, 1.0], [0.0, -3.0]],
                   [[-2.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], [[1.0, 0.0], [-5.0, 0.0], [1.0, 0.0]],
                   [[0.0, -1.0], [-3.0, 0.0], [0.0, 2.0]], [[-3.0, 0.0], [0.0, -1.0], [0.0, 3.0]],
                   [[0.0, 2.0], [-3.0, 0.0], [0.0, 1.0]], [[-2.0, 0.0], [0.0, 4.0], [-1.0, 0.0]],
                   [[-1.0, 0.0], [0.0, -4.0], [-1.0, 0.0]], [[1.0, 0.0], [-3.0, 0.0], [0.0, -3.0]],
                   [[2.0, 0.0], [0.0, -2.0], [-4.0, 0.0]], [[1.0, 0.0], [0.0, -1.0], [-3.0, 0.0]],
                   [[0.0, -2.0], [-2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [-4.0, 0.0], [0.0, 1.0]],
                   [[-2.0, 0.0], [0.0, -2.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, 3.0], [-3.0, 0.0]],
                   [[1.0, 0.0], [-3.0, 0.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, -4.0], [-2.0, 0.0]],
                   [[-2.0, 0.0], [0.0, -3.0], [1.0, 0.0]], [[0.0, -4.0], [-1.0, 0.0], [0.0, 2.0]],
                   [[2.0, 0.0], [-3.0, 0.0], [0.0, -1.0]], [[0.0, -3.0], [-1.0, 0.0], [0.0, 3.0]],
                   [[0.0, -2.0], [-1.0, 0.0], [0.0, 3.0]], [[-2.0, 0.0], [0.0, 2.0], [1.0, 0.0]],
                   [[0.0, 1.0], [-1.0, 0.0], [0.0, 2.0]], [[0.0, -1.0], [-1.0, 0.0], [0.0, 5.0]],
                   [[0.0, -2.0], [0.0, 1.0], [0.0, -3.0]], [[0.0, -1.0], [0.0, 1.0], [0.0, -3.0]],
                   [[-1.0, 0.0], [0.0, -2.0], [1.0, 0.0]], [[2.0, 0.0], [0.0, -1.0], [-2.0, 0.0]],
                   [[-2.0, 0.0], [0.0, 1.0], [2.0, 0.0]], [[-3.0, 0.0], [3.0, 0.0], [0.0, 2.0]],
                   [[1.0, 0.0], [0.0, 3.0], [-1.0, 0.0]], [[-2.0, 0.0], [2.0, 0.0], [0.0, 4.0]],
                   [[-1.0, 0.0], [0.0, -4.0], [2.0, 0.0]], [[0.0, -1.0], [1.0, 0.0], [0.0, -2.0]],
                   [[1.0, 0.0], [0.0, -4.0], [0.0, 2.0]], [[0.0, 3.0], [1.0, 0.0], [0.0, -4.0]],
                   [[0.0, -3.0], [0.0, 3.0], [1.0, 0.0]], [[0.0, 1.0], [4.0, 0.0], [-3.0, 0.0]],
                   [[-1.0, 0.0], [0.0, 2.0], [2.0, 0.0]], [[-1.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
                   [[0.0, 4.0], [-1.0, 0.0], [2.0, 0.0]], [[-1.0, 0.0], [3.0, 0.0], [0.0, -4.0]],
                   [[0.0, -4.0], [2.0, 0.0], [0.0, 1.0]], [[0.0, -1.0], [2.0, 0.0], [0.0, -1.0]],
                   [[3.0, 0.0], [-1.0, 0.0], [0.0, -1.0]], [[-2.0, 0.0], [5.0, 0.0], [-1.0, 0.0]],
                   [[3.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], [[0.0, 2.0], [-2.0, 0.0], [4.0, 0.0]],
                   [[0.0, 3.0], [-1.0, 0.0], [3.0, 0.0]], [[0.0, 3.0], [2.0, 0.0], [0.0, 1.0]],
                   [[0.0, -2.0], [3.0, 0.0], [0.0, -2.0]], [[1.0, 0.0], [0.0, -3.0], [2.0, 0.0]],
                   [[0.0, -3.0], [0.0, 1.0], [3.0, 0.0]], [[-1.0, 0.0], [4.0, 0.0], [0.0, -1.0]],
                   [[2.0, 0.0], [-2.0, 0.0], [3.0, 0.0]], [[-1.0, 0.0], [0.0, 1.0], [4.0, 0.0]],
                   [[1.0, 0.0], [0.0, 2.0], [2.0, 0.0]], [[0.0, 4.0], [3.0, 0.0], [0.0, -1.0]],
                   [[0.0, 2.0], [3.0, 0.0], [0.0, 2.0]], [[0.0, -2.0], [4.0, 0.0], [0.0, -2.0]],
                   [[2.0, 0.0], [0.0, -3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, -2.0], [3.0, 0.0]],
                   [[0.0, 1.0], [0.0, -2.0], [4.0, 0.0]], [[0.0, -1.0], [4.0, 0.0], [0.0, 1.0]],
                   [[0.0, 1.0], [-1.0, 0.0], [5.0, 0.0]], [[0.0, 2.0], [-1.0, 0.0], [5.0, 0.0]],
                   [[2.0, 0.0], [0.0, 3.0], [2.0, 0.0]], [[0.0, 2.0], [4.0, 0.0], [0.0, 2.0]]]

p1 = []
for i0 in range(len(trained_targets)):
    p1.append(1)
print(p1)
p1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
trained_target_probs = []
for i0 in range(len(trained_targets)):
    trained_target_probs.append(0)
trained_target_probs.append(1)
print(trained_target_probs)
trained_target_probs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

list_len = []
fullstep_counts_list = []
for i0 in range(len(trained_targets)):
    list_len.append(np.sum(abs(np.array(trained_targets[i0]))))
    fullstep_counts_list.append(len(trained_targets[i0]))
list_len = np.array(list_len)
fullstep_counts_list = np.array(fullstep_counts_list)
group_len_count = np.array(list(set(zip(fullstep_counts_list, list_len))))
list_len_count = np.array(list(zip(fullstep_counts_list, list_len)))
alpha_group_id_list = np.zeros(len(trained_targets), dtype=int)
for i0 in range(len(trained_targets)):
    if fullstep_counts_list[i0] == 1:
        alpha_group_id_list[i0] = 0
        continue
    alpha_group_id_list[i0] = int(np.where(np.sum(list_len_count[i0, :] == group_len_count, axis=1) == 2)[0][0])
alpha_group_id_list = list(alpha_group_id_list)
print(alpha_group_id_list)
alpha_list = list(np.ones(group_len_count.shape[0], dtype=int))
print(alpha_list)

step_count_list1 = []
for i0 in range(len(fullstep_counts_list)):
    if fullstep_counts_list[i0] == 1:
        step_count_list1.append(1)
    else:
        if i0 % 2 == 0:
            step_count_list1.append(1)
        else:
            step_count_list1.append(fullstep_counts_list[i0])
fullstep_counts_list = list(fullstep_counts_list)
print(fullstep_counts_list)

###
validation_targets = [[[0, -2]], [[0, -2], [2, 0]], [[0, -2], [-4, 0]], [[0, -2], [2, 0], [-4, 0]]]
rep = 10

for i0 in range(len(validation_targets)):
    trained_targets.append(validation_targets[i0])
    p1.append(rep)
    trained_target_probs.insert(-1, 0)
    alpha_group_id_list.append(max(alpha_group_id_list)+1)
    alpha_list.append(0)
    fullstep_counts_list.append(len(validation_targets[i0]))

oneshot_counts_list = list(np.ones(len(p1)).astype(int))

print(trained_targets)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)
print(oneshot_counts_list)
print(fullstep_counts_list)
pass
