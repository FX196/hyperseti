ncpu = 40
from hyperseti import hyperseti
import sys

for i in [1, 2, 3, 4]:
    nranks, ngpu = i * 4, i
    sys.stderr.write(str(i) + "\n")
    print(f"###### {i} ######")
    hyperseti.find_et_parallel("GBT_58210_31246_HIP93185_fine.h5", nranks=nranks, ngpu=ngpu, gulp_size=2**12, ngulps=120 * i + nranks, freq_start=116048342, apply_normalization=True)