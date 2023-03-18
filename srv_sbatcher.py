import argparse
import os
import socket
import time
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='minisbatcher', allow_abbrev=False)
    parser.add_argument('--file', type=str, required=True,)
    parser.add_argument('--at_a_time', type=int, default=-1)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--skip_first', type=int, default=0)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--force_dp', type=int, default=0)
    parser.add_argument('--name', type=str, default="mammoth")
    parser.add_argument('--mem', type=int)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--sync_after', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--timelimit', type=str, default="1-0")
    # parser.add_argument('--silendo', action='store_true')
    parser.add_argument('--per_job', type=int, default=1)
    parser.add_argument('--excludelist', type=str, default=None)
    parser.add_argument('--cpus', type=int)

    args = parser.parse_args()

    args.silendo = 'GLADIO' in os.environ and os.environ.get('GLADIO') == "p2"
    args.cineca = "cineca" in socket.getfqdn().lower()
    
    assert not (args.silendo and args.cineca)

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

    def bbasename(path):
        return [x for x in path.split('/') if len(x)][-1]
    conf_path = os.getcwd()

    errbase, outbase = 'err', 'out'
    if args.silendo:
        while not 'gladio' in bbasename(os.path.dirname(conf_path)):
            conf_path = os.path.dirname(conf_path)

        uname = conf_path.split('/')[-1]
        assert os.path.exists(f'/shared/{uname}'), f"Create /shared/{uname} with chmod 777 permissions first!"
        os.makedirs(f'/shared/{bbasename(conf_path)}/err', exist_ok=True)
        os.makedirs(f'/shared/{bbasename(conf_path)}/out', exist_ok=True)
        os.makedirs(f'/shared/{bbasename(conf_path)}/sbfiles', exist_ok=True)
        errbase, outbase = f'/shared/{bbasename(conf_path)}/err', f'/shared/{bbasename(conf_path)}/out'

    if not os.path.exists(errbase):
        os.makedirs(errbase)
    if not os.path.exists(outbase):
        os.makedirs(outbase)

    tdir = os.getcwd()
    # os.environ['PYTHONPATH'] = f'{conf_path}'
    # os.environ['PATH'] += f':{conf_path}'
    # os.chdir(conf_path)
    len_com = math.ceil(len(all_com) / args.per_job)
    python = 'python'
    if not args.silendo and not args.cineca and os.environ['LOGNAME'] == 'efrascaroli':
        python = '/homes/efrascaroli/.conda/envs/SRV-Continual/bin/python'
    jobstring =  ' &\nsleep 60s; '.join([f'{python} utils/main.py ${{args[$(($SLURM_ARRAY_TASK_ID * {args.per_job} + {i}))]}}' for i in range(args.per_job)])
    exclusion = '' if args.excludelist is None else '#SBATCH --exclude=' + args.excludelist
    all_com_str = "".join([f"' {s} '\n" for s in all_com]).strip()
    filec = f"""#!/bin/bash
{"#SBATCH -p m100_usr_prod" if args.cineca else "#SBATCH -p prod"}
#SBATCH --job-name={args.name}
{"#SBATCH --nodes=1" if not args.silendo else ""}
#SBATCH --time={args.timelimit}
{f"#SBATCH --mem={args.mem}G" if args.mem else ""}
#SBATCH --output="{os.path.join(outbase, args.name + r'_%A_%a.out')}"
#SBATCH --error="{os.path.join(errbase, args.name + r'_%A_%a.out')}"
{"#SBATCH -A IscrB_LEGOCEMM" if args.cineca else "#SBATCH -A ricerca_generica" if not args.silendo else ""}
#SBATCH --gres=gpu:{args.gpus}
{f"#SBATCH --cpus-per-task={args.cpus}" if args.cpus is not None else ""}
#SBATCH --array=0-{len_com-1}%{(len_com if args.at_a_time <= 0 else args.at_a_time)}
{exclusion}

args=(
{all_com_str}
)
export PYTHONPATH={os.getcwd()}
cd {os.getcwd()}
{'export WANDBBQ_RELAY=login02' if args.cineca else ''}

""" + jobstring + '\nwait'

    outpath = 'mini_sbatch.sh'
    if args.silendo:
        outpath = f'/shared/{bbasename(conf_path)}/sbfiles/{args.name}.sh'

    with open(outpath, "w") as f:
        f.write(filec)
    if args.dry:
        print(f'check {outpath}')
        exit(0)
    jobid = os.popen(f'sbatch {outpath}').read().splitlines()[-1].split()[-1].strip()

    if args.sync_after and not args.dry and not args.silendo:
        filec=f"""#!/bin/bash
#SBATCH --job-name={args.name}
#SBATCH --nodes=1
#SBATCH --output="out/sync_%j.out"
#SBATCH --error="err/sync_%j.out"
#SBATCH --dependency=singleton
{"#SBATCH -A IscrB_LEGOCEMM" if args.cineca else "#SBATCH -A ricerca_generica"}
{"#SBATCH -p m100_all_serial" if args.cineca else ""}

{"conda activate SRV-Continual" if not args.cineca and not args.silendo else ""}
cat {errbase}/*{jobid}* | grep 'wandb sync' | sed 's/^wandb: //g' | xargs -i bash -c {{}}
"""

        with open("mini_sync.sh", "w") as f:
            f.write(filec)
        os.system(f'sbatch mini_sync.sh')
