import argparse
import os
import socket
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='minisbatcher', allow_abbrev=False)
    parser.add_argument('--file', type=str, required=True,)
    parser.add_argument('--at_a_time', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--skip_first', type=int, default=0)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--force_dp', type=int, default=0)
    parser.add_argument('--name', type=str, default="mammoth")
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--mem', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--account', type=str, default="FF4_Axyon", choices=["IscrC_ADELE-CL","FF4_Axyon"])
    parser.add_argument('--sync_after', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        all_com = f.read().splitlines()
    all_com = [x for x in all_com if not x.startswith('#') and len(x.strip())]
    all_com = all_com * args.cycles

    if args.reverse:
        all_com = all_com[::-1]
    all_com = all_com[args.skip_first:]
    if args.debug:
        sss = []
        for s in all_com:
            vv = " ".join(["--n_epochs=1" if "n_epochs=" in c else c for c in s.split()] + [" --debug_mode=1"])
            sss.append(vv)
        all_com = sss
        args.name="debug_"+args.name

    all_com_str = "".join([f"' {s} '\n" for s in all_com]).strip()
    filec = f"""#!/bin/bash
#SBATCH -p m100_usr_prod
#SBATCH --job-name={args.name}
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mem={args.mem}G
#SBATCH --account={args.account}

#SBATCH --output="out/{args.name}_%A_%a.out"
#SBATCH --error="err/{args.name}_%A_%a.out"
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --array=0-{len(all_com)-1}%{len(all_com)}

args=(
{all_com_str}
)

""" + "python utils/main.py ${args[$SLURM_ARRAY_TASK_ID]}"

    with open("mini_sbatch.sh", "w") as f:
        f.write(filec)
    if args.dry:
        print('check "mini_sbatch.sh"')
        exit(0)
    os.system(f'sbatch mini_sbatch.sh')

    if args.sync_after and not args.dry:
        filec=f"""#!/bin/bash
#SBATCH -p m100_all_serial
#SBATCH --job-name={args.name}
#SBATCH --nodes=1
#SBATCH --output="out/sync_%j.out"
#SBATCH --error="err/sync_%j.out"
#SBATCH --dependency=singleton
wandb sync --sync-all"""

    with open("mini_sync.sh", "w") as f:
        f.write(filec)
    os.system(f'sbatch mini_sync.sh')
