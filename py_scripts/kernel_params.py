import subprocess
import os
# Set the working directory
os.chdir('/home/ubuntu/Documents/pa1-haw006-jiw015')

N_DOUBLES_IN_L1 = 2**13  # 64*1024/8 = 8192
N_DOUBLES_IN_L2 = 2**17  # 1024^2/8 = 128*1024
# L3 is 32 MB, big enough

# Define lists of possible values for each macro
## still loop for proper message, see below
DGEMM_NR_values = [4]
## {8} in define BL_MICRO_KERNEL bl_dgemm_{8}44 is MR, 
## and makefile will take care of it
DGEMM_MR_values = [8, 16]
DGEMM_KC_values = [64, 128, 256, 512]
DGEMM_NC_values = [64, 128, 256, 512]
DGEMM_MC_values = [512, 1024]

# run and store (append) result to txt
output_file = "results/kernel_params_raw_result.txt"
benchmark_command = f"./benchmark-blislab | grep 'GeoMean' >> {output_file}"

# file title
subprocess.run(f"echo 'NEW RUN' > {output_file}", shell=True, check=True)

# Iterate through combinations of macro values
for nr in DGEMM_NR_values:
    for mr in DGEMM_MR_values:
        for kc in DGEMM_KC_values:
            if kc * mr >= N_DOUBLES_IN_L1:
                continue
            for nc in DGEMM_NC_values:
                if kc * nc >= N_DOUBLES_IN_L2:
                    continue
                for mc in DGEMM_MC_values:
                    # divider
                    divider = f"{nr}\t{mr}\t{kc}\t{nc}\t{mc}"
                    subprocess.run(f"echo '{divider}' >> {output_file}", shell=True, check=True)

                    # Construct the make command with current macro values
                    make_command = [
                        'make',
                        '-s',
                        f'DGEMM_NR={nr}',
                        f'DGEMM_MR={mr}',
                        f'DGEMM_KC={kc}',
                        f'DGEMM_NC={nc}',
                        f'DGEMM_MC={mc}'
                    ]

                    # Execute the make command and wait for completion
                    subprocess.run(
                        make_command, 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )

                    # Execute the compiled executable and wait for completion
                    # As suggested by the Writeup, run 3 times for each (to take highest value)
                    subprocess.run(benchmark_command, shell=True, stdout=subprocess.DEVNULL, check=True)
                    subprocess.run(benchmark_command, shell=True, stdout=subprocess.DEVNULL, check=True)
                    subprocess.run(benchmark_command, shell=True, stdout=subprocess.DEVNULL, check=True)

                    print(f"DONE for NR={nr}, MR={mr}, KC={kc}, NC={nc}, MC={mc}")
