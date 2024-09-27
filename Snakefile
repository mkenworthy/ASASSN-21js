rule get200samples:
    input:
        "src/data/mcmc_20240208_backup_2e6.h5",
        "src/data/a21js_Vgf_avgsqrt_bin9days.csv"
    output:
        "src/data/rand200_mcmc_20240208_backup_2e6.pkl"
    conda:
        "environment.yml"
    script:
        "src/scripts/Generate_Random_200.py"

rule make9daybin:
    input:
        "src/data/ASASSN_lc_new230124.csv"
    output:
        "src/data/a21js_Vgf_avgsqrt_bin9days.csv"
    conda:
        "environment.yml"
    script:
        "src/scripts/Data_Preprocessing_Clean.py"