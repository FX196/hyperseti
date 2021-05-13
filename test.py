from hyperseti import hyperseti

hyperseti.find_et_parallel("GBT_58210_31246_HIP93185_fine.h5", nranks=40, gulp_size=2**15, ngulps=40, freq_start=116048342, apply_normalization=True)
