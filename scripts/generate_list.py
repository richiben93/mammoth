import os
import sys
from argparse import ArgumentParser, Namespace
from random import randint

from typing import List

#--dataset seq-cifar100-10x10 --wandb --model er_ace_replay --lr 0.1 --minibatch_size 64 --batch_size 64 --n_epochs 20 --buffer_size 2000 --rep_minibatch 128 --knn_laplace 20 --replay_weight 0.1      --replay_mode none

#--dataset seq-cifar100-10x10 --model joint_replay --batch_size 128 --lr 0.1 --n_epochs 1 --buffer_size 2000 --minibatch_size 64 --rep_minibatch 512 --knn_laplace 20 --fmap_dim 120 --replay_mode egap3 --replay_weight 0.1

grid = {
    "dataset": ["seq-cifar100-10x10"],  # "seq-cifar100-10x10"],
    "wandb": [True],
    "model": ["er_ace_replay"],  # "joint_replay", "er_ace_replay"],
    "batch_size": ["64"],
    "lr": ["0.1"],
    "n_epochs": ["50"],
    "lr_decay_steps": ["35,45"],
    "buffer_size": ["2000"],
    "minibatch_size": ["64"],
    "rep_minibatch": ["512"],
    "heat_kernel": [],
    "knn_laplace": ["20"],
    "fmap_dim": ["160"],
    "replay_mode": ["none"],    # egap2 egap2-1 egap3
    "cos_dist": [],
    "replay_weight": ["0.1"],
    "save_checks": [],
    "custom_log": [],
}
#--output sbatch/mammoth/list_sbatch.txt --title "ejoint" --comment_previous
props = {
    "output": "sbatch/mammoth/list_sbatch.txt",
    "title": "",
    "comment_previous": True,
}


# "   ".rjust(len(key), ' ')
def get_value(grid_dict: dict, key: str, idx: int, span_len=0, prefix=' --'):
    if type(grid_dict[key][idx]) == bool:
        return prefix + key if grid_dict[key][idx] else " " * (len(prefix) + len(key))
    val = prefix + key + " " + grid_dict[key][idx]
    if len(val) < span_len:
        val = val.ljust(span_len)
    return val


lines: List[str] = [""]
for key, values in grid.items():
    if len(values) == 0:
        continue
    if len(values) == 1:
        str_value = get_value(grid, key, 0)
        for i in range(len(lines)):
            lines[i] += str_value
        continue
    max_len = 0
    for i in range(len(values)):
        cur_len = len(get_value(grid, key, i))
        if cur_len > max_len:
            max_len = cur_len
    new_lines = []
    for line in lines:
        for i in range(len(values)):
            str_value = get_value(grid, key, i, max_len)
            new_lines.append(line + str_value)
    lines = new_lines
    # print(values)

# for i in range(len(lines)):
#     lines[i] = lines[i][1:]
print(f"{len(lines)} lines generated")

# parser = ArgumentParser(description='Generate list for srv-sbatch', allow_abbrev=False)
# parser.add_argument('--output', type=str, required=False, help='File in which to insert sbatch lines (default stdout).')
# parser.add_argument('--title', type=str, required=False, help='Add title (comment in output).')
# parser.add_argument('--comment_previous', action='store_true', help='Comment previous lines in output file.')
# args = parser.parse_args(props)
args = Namespace(**props)

if args.title is not None:
    lines.insert(0, f"## {args.title}")
if args.output is None:
    args.comment_previous = False
    print("\n".join(lines))
else:
    if args.comment_previous and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            prev_lines = [line for line in f.readlines()]
        with open(args.output, 'w') as f:
            for line in prev_lines:
                f.write(line if line.startswith("#") or line == '\n' else "#"+line)
    with open(args.output, 'a') as f:
        f.write("\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"{len(lines)} lines written to {args.output}")

# print(vars(args))

# with open(args.output) as f:
#     sbacci = f.read().splitlines()

# with open(args.output, 'w') as f:
#     f.write("gridsh")
