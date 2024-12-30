import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import *
import copy


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
ends_all = np.delete(starts_all, np.where(np.sum(starts_all == 0, axis=1) == 2), axis=0)

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

training_set = []

two_step_moves_list = []
for i0 in range(start_moves.shape[0]):
    two_step_moves_list.append(start_moves[i0].tolist())

idx_pick_list = []
for x in range(-3, 4):
    for y in range(-3, 4):
        idxs_pick = np.where(np.sum(start_move_ends == [x, y], axis=1) == 2)[0]
        if idxs_pick.shape[0] != 0:
            idx_pick = int(np.random.choice(idxs_pick, 1))

            training_set.append(two_step_moves_list[idx_pick])
            idx_pick_list.append(idx_pick)
        else:
            a = 1

moves = [training_set]
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
    alpha_group_id_list[i0] = int(list_len[i0] - 1)
alpha_group_id_list = list(alpha_group_id_list)
alpha_list = list(np.ones(len(np.unique(np.array(alpha_group_id_list))), dtype=int))

# print(trained_targets)
# print(p1)
# print(trained_target_probs)
# print(alpha_group_id_list)
# print(alpha_list)

###
training_vectors = []
[training_vectors.append(training_set[i0][1]) for i0 in range(len(training_set))]
vector_dict = {}
for i0 in range(acts_all.shape[0]):
    vector_dict[tuple(acts_all[i0])] = 0

for v in training_vectors:
    vector_dict[tuple(v)] += 1

training_starts = []
[training_starts.append(training_set[i0][0]) for i0 in range(len(training_set))]
starts_dict = {}
for i0 in range(starts_all.shape[0]):
    starts_dict[tuple(starts_all[i0])] = 0

for s in training_starts:
    starts_dict[tuple(s)] += 1

missing_start = []
for k, v in starts_dict.items():
    if v == 0:
        missing_start.append(k)

missing_vector = []
for k, v in vector_dict.items():
    if v == 0:
        missing_vector.append(k)

i = 0
while i < len(missing_vector):
    v = missing_vector[i]
    for s in missing_start:
        start_move_ = np.array([list(s), list(v)])[np.newaxis, :, :]
        if filter(start_move_).shape[0] != 0:
            missing_vector.remove(v)
            missing_start.remove(s)
            training_set.append([list(s), list(v)])
            i -= 1
            break

    i += 1

training_ends = []
[training_ends.append(np.array(training_set[i0][0]) + np.array(training_set[i0][1])) for i0 in range(len(training_set))]
ends_dict = {}
for i0 in range(ends_all.shape[0]):
    ends_dict[tuple(ends_all[i0])] = 0

for s in training_ends:
    ends_dict[tuple(s)] += 1

# rank ends frequency from low to high
ends_key = ends_dict.keys()
ends_val = ends_dict.values()

idx_sort = np.argsort(list(ends_val))

i = 0
while i < len(missing_start):
    s = np.array(missing_start[i])
    candidate_s_m = filter(np.stack((np.repeat(s[np.newaxis, :], acts_all.shape[0], axis=0), acts_all), axis=1))
    idx = np.random.choice(candidate_s_m.shape[0], 1)
    training_set.append([list(candidate_s_m[idx, 0, :][0]), list(candidate_s_m[idx, 1, :][0])])
    missing_start.remove(missing_start[i])

trained_targets = training_set

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
    alpha_group_id_list[i0] = int(list_len[i0] - 1)
alpha_group_id_list = list(alpha_group_id_list)
alpha_list = list(np.ones(len(np.unique(np.array(alpha_group_id_list))), dtype=int))

# print(trained_targets)
# print(p1)
# print(trained_target_probs)
# print(alpha_group_id_list)
# print(alpha_list)
additional_training_set = [[[1, 1], [-2, 0]], [[1, 1], [0, -2]], [[-1, 1], [0, -2]], [[-1, 1], [2, 0]],
                           [[-1, -1], [2, 0]], [[-1, -1], [0, 2]], [[1, -1], [-2, 0]], [[1, -1], [0, 2]]]
# set 1
test_targets_1 = [[[1, 1], [-2, 0]], [[1, 1], [0, -2]], [[-1, 1], [0, -2]], [[-1, 1], [2, 0]],
                  [[-1, -1], [2, 0]], [[-1, -1], [0, 2]], [[1, -1], [-2, 0]], [[1, -1], [0, 2]], [[1, 1], [2, 0]],
                  [[1, 1], [0, 2]],
                  [[-1, 1], [-2, 0]], [[-1, 1], [0, 2]],
                  [[-1, -1], [-2, 0]], [[-1, -1], [0, -2]],
                  [[1, -1], [0, -2]], [[1, -1], [2, 0]],
                  [[-1, 3], [0, -2]], [[1, 3], [0, -2]], [[3, 1], [-2, 0]], [[3, -1], [-2, 0]], [[1, -3], [0, 2]],
                  [[-1, -3], [0, 2]], [[-3, -1], [2, 0]], [[-3, 1], [2, 0]]]

# set 2
test_targets_2 = [[[-3, 2], [1, 0]], [[-3, 2], [3, 0]], [[-3, 2], [6, 0]], [[0, 2], [3, 0]], [[2, 2], [1, 0]],
                  [[2, 3], [0, -1]], [[2, 3], [0, -3]], [[2, 3], [0, -6]], [[2, 0], [0, -3]], [[2, -2], [0, -1]],
                  [[3, -2], [-1, 0]], [[3, -2], [-3, 0]], [[3, -2], [-6, 0]], [[0, -2], [-3, 0]], [[-2, -2], [-1, 0]],
                  [[-2, -3], [0, 1]], [[-2, -3], [0, 3]], [[-2, -3], [0, 6]], [[-2, 0], [0, 3]], [[-2, 2], [0, 1]]]

# set 3 random
length_set = np.tile(np.arange(1, 7), 4)
length_set = np.flip(np.sort(length_set))
direction_set = np.tile(np.array([[0, 1], [0, -1], [-1, 0], [1, 0]]), (6, 1))
end_set = np.array([[-2, 3], [0, 3], [2, 3], [-3, 2], [-1, 2], [1, 2], [3, 2],
                    [-2, 1], [0, 1], [2, 1], [-3, 0], [-1, 0], [1, 0], [3, 0],
                    [-2, -1], [0, -1], [2, -1], [-3, -2], [-1, -2], [1, -2], [3, -2],
                    [-2, -3], [0, -3], [2, -3]])

test_targets_3 = []

while length_set.shape[0] != 0:
    idx_l = 0
    idx_d = np.random.choice(direction_set.shape[0], 1)
    idx_e = np.random.choice(end_set.shape[0], 1)
    v = length_set[idx_l] * direction_set[idx_d, :] * -1
    start_vec = np.stack((end_set[idx_e][0], v[0]))[np.newaxis, :, :]
    while (filter(start_vec).shape[0] == 0):
        idx_d = np.random.choice(direction_set.shape[0], 1)
        idx_e = np.random.choice(end_set.shape[0], 1)
        v = length_set[idx_l] * direction_set[idx_d, :] * -1
        start_vec = np.stack((end_set[idx_e][0], v[0]))[np.newaxis, :, :]

    test_targets_3.append([list(end_set[idx_e][0] + v[0]), list(-v[0])])
    length_set = np.delete(length_set, 0)
    direction_set = np.delete(direction_set, idx_d, axis=0)
    end_set = np.delete(end_set, idx_e, axis=0)

## new training set for loop
trained_targets_loop = copy.deepcopy(trained_targets)
overlap_loop = []
for i0 in range(len(trained_targets)):
    tar_i0 = trained_targets[i0]
    if (tar_i0 in additional_training_set):
        overlap_loop.append(tar_i0)
        trained_targets_loop.remove(tar_i0)

target_set_loop = copy.deepcopy(trained_targets_loop)
[target_set_loop.append(i0) for i0 in additional_training_set]

p1 = []
for i0 in range(len(target_set_loop)):
    if i0 < len(trained_targets_loop):
        p1.append(1)
    else:
        p1.append(10)

trained_target_probs = []
for i0 in range(len(target_set_loop)):
    trained_target_probs.append(0)
trained_target_probs.append(1)

alpha_group_id_list = []

for i0 in range(len(target_set_loop)):
    if i0 < len(trained_targets_loop):
        alpha_group_id_list.append(0)
    else:
        alpha_group_id_list.append(1)
alpha_list = [1, 1]

print(target_set_loop)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)

## testing set 1
trained_targets_1 = copy.deepcopy(trained_targets)
test_ends_1 = np.sum(np.array(test_targets_1), axis=1).tolist()
overlap_pair1 = []
overlap_end1 = []
for i0 in range(len(trained_targets)):
    tar_i0 = trained_targets[i0]
    end_i0 = [tar_i0[0][0] + tar_i0[1][0], tar_i0[0][1] + tar_i0[1][1]]
    if (tar_i0 in test_targets_1) or (end_i0 in test_ends_1):
        if tar_i0 in test_targets_1:
            overlap_pair1.append(tar_i0)
        if end_i0 in test_ends_1:
            overlap_end1.append(tar_i0)
        trained_targets_1.remove(tar_i0)

target_set1 = copy.deepcopy(trained_targets_1)
[target_set1.append(i0) for i0 in test_targets_1]

p1 = []
for i0 in range(len(target_set1)):
    if i0 < len(trained_targets_1):
        p1.append(1)
    else:
        p1.append(6)

trained_target_probs = []
for i0 in range(len(target_set1)):
    trained_target_probs.append(0)
trained_target_probs.append(1)

alpha_group_id_list = []

for i0 in range(len(target_set1)):
    if i0 < len(trained_targets_1):
        alpha_group_id_list.append(0)
    else:
        alpha_group_id_list.append(1)
alpha_list = [1, 0]

print(target_set1)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)

### testing set 2
trained_targets_2 = copy.deepcopy(trained_targets)
test_ends_2 = np.sum(np.array(test_targets_2), axis=1).tolist()
overlap_pair2 = []
overlap_end2 = []
for i0 in range(len(trained_targets)):
    tar_i0 = trained_targets[i0]
    end_i0 = [tar_i0[0][0] + tar_i0[1][0], tar_i0[0][1] + tar_i0[1][1]]
    if (tar_i0 in test_targets_2) or (end_i0 in test_ends_2):
        if tar_i0 in test_targets_2:
            overlap_pair2.append(tar_i0)
        if end_i0 in test_ends_2:
            overlap_end2.append(tar_i0)
        trained_targets_2.remove(tar_i0)

target_set2 = copy.deepcopy(trained_targets_2)
[target_set2.append(i0) for i0 in test_targets_2]

p1 = []
for i0 in range(len(target_set2)):
    if i0 < len(trained_targets_2):
        p1.append(1)
    else:
        p1.append(7)

trained_target_probs = []
for i0 in range(len(target_set2)):
    trained_target_probs.append(0)
trained_target_probs.append(1)

alpha_group_id_list = []

for i0 in range(len(target_set2)):
    if i0 < len(trained_targets_2):
        alpha_group_id_list.append(0)
    else:
        alpha_group_id_list.append(1)
alpha_list = [1, 0]

print(target_set2)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)


### testing set 3
trained_targets_3 = copy.deepcopy(trained_targets)
test_ends_3 = np.sum(np.array(test_targets_3), axis=1).tolist()
overlap_pair3 = []
overlap_end3= []
for i0 in range(len(trained_targets)):
    tar_i0 = trained_targets[i0]
    end_i0 = [tar_i0[0][0] + tar_i0[1][0], tar_i0[0][1] + tar_i0[1][1]]
    if (tar_i0 in test_targets_3) or (end_i0 in test_ends_3):
        if tar_i0 in test_targets_3:
            overlap_pair3.append(tar_i0)
        if end_i0 in test_ends_3:
            overlap_end3.append(tar_i0)
        trained_targets_3.remove(tar_i0)

target_set3 = copy.deepcopy(trained_targets_3)
[target_set3.append(i0) for i0 in test_targets_3]

p1 = []
for i0 in range(len(target_set3)):
    if i0 < len(trained_targets_3):
        p1.append(1)
    else:
        p1.append(4)

trained_target_probs = []
for i0 in range(len(target_set3)):
    trained_target_probs.append(0)
trained_target_probs.append(1)

alpha_group_id_list = []

for i0 in range(len(target_set3)):
    if i0 < len(trained_targets_3):
        alpha_group_id_list.append(0)
    else:
        alpha_group_id_list.append(1)
alpha_list = [1, 0]

print(target_set3)
print(p1)
print(trained_target_probs)
print(alpha_group_id_list)
print(alpha_list)
a = 1
