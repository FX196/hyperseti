from hyperseti import hyperseti
import sys

for i in [1, 2, 3, 4, 8, 16, 32, 40]:
    sys.stderr.write(str(i) + "\n")
    print(f"###### {i} ######")
    hyperseti.find_et_parallel("GBT_58210_31246_HIP93185_fine.h5", nranks=i, ngpu=4, max_dd=5, threshold=25, gulp_size=2**18, ngulps=160+i, freq_start=116048342, apply_normalization=True)


# for i in [1, 2, 3, 4]:
#     sys.stderr.write(str(i) + "\n")
#     print(f"###### {i} ######")
#     hyperseti.find_et_parallel("GBT_58210_31246_HIP93185_fine.h5", nranks=40, ngpu=i, gulp_size=2**12, ngulps=480, freq_start=116048342, apply_normalization=True)