import matplotlib.pyplot as plt
import re
import numpy as np

weak_scale = True
strong_scale_fix_cpu = False
strong_scale_fix_gpu = False

if weak_scale:
    filename = "weak_scale"
elif strong_scale_fix_cpu:
    filename = "strong_scale_fix_cpu"
else:
    filename = "strong_scale_fix_gpu"

with open(filename, "r") as f:
    results = f.read().split("\n")

nranks, times = [], []
ngpus = []
time_map = {}
for line in results:
    if "pipeline times" in line:
        print(line.split())
        tokens = line.split()
        times = eval(''.join(tokens[6:-3]))[1:] # exclude first element
        nrank = int(tokens[3])
        nranks.append(nrank)
        ngpu = int(tokens[1])
        ngpus.append(ngpu)
        if strong_scale_fix_cpu:
            if ngpu not in time_map:
                time_map[ngpu] = []
            time_map[ngpu].append(sum(times))
        else:
            if nrank not in time_map:
                time_map[nrank] = []
            time_map[nrank].append(sum(times))
        # print(tokens)
        # times.append(float(tokens[-2].strip('s')))
        # nranks.append(tokens[4])
        # ngpus.append(tokens[6])
for key, val in time_map.items():
    time_map[key] = np.mean(val)

if weak_scale:
    nranks = list(time_map.keys())
    times = list(time_map.values()) 
    ax = plt.subplot() 
    ax.set_xticks(nranks)
    ticklabels = [f'{rank} CPU(s)/{rank//8} GPU(s)' for rank in nranks]
    ax.set_xticklabels(ticklabels)
    print(nranks)
    
    ax.plot(nranks, times, color='black', marker='o')
    ax.set_xlabel("Number of GPUs/CPUs used")
    ax.set_ylabel("Total runtime excluding first run (s)")
    ax.set_ylim((0, 30))
    plt.savefig("weak_scale.png")
elif strong_scale_fix_cpu:
    gpus = list(time_map.keys())
    times = list(time_map.values()) 
    ax = plt.subplot()
    ax.plot(gpus, times, color='black', marker='o')
    # ax.set_ylim((0, 80))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of GPUs used")
    ax.set_ylabel("Total runtime excluding first run (s)")
    plt.savefig("fix_cpu_strong_scale.png")
else:
    nranks = list(time_map.keys())
    times = list(time_map.values())  
    ax = plt.subplot()
    ax.plot(nranks, times, color='black', marker='o')
    ax.set_xlabel("Number of CPUs used")
    ax.set_ylabel("Total runtime excluding first run (s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("fix_gpu_strong_scale.png")
# nranks = list(time_map.keys())
# times = list(time_map.values())  
# ax = plt.subplot()
# ax.plot(nranks, times, color='black', marker='o')
# ax.set_xlabel("Number of CPUs used")
# ax.set_ylabel("Average pipeline time")
# plt.savefig("fix_gpu_strong_scale.png")

# plt.plot(gpus, times, color='black', marker='o', ylim=(0, 5))
# plt.xlabel("Number of GPUs used")
# plt.ylabel("Total runtime")
# plt.savefig("fix_cpu_strong_scale.png")

