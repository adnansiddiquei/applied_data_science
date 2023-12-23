# M1: Applied Data Science - Coursework Assignment
## Table of Contents
1. [Running the code](#run)
2. [Report](#report)

## <a name="run"></a> 1. Running the code
Clone the repository and then run
```bash
cd as3438
make run
```

This will
1. Build the Docker image using the [Dockerfile](Dockerfile).
2. Run the Docker image, and mount all the output folders so the figures and outputs are saved to your local machine.
3. Open an interactive terminal in the Docker container.

Then activate the conda environment by running
```bash
conda activate as3438_m1cw
```
and then you can run all the scripts
```bash
python -m src
```

The terminal will print updates on the programs progress as it works through the questions.

After each question finishes running, you will find all of the outputs in the relevant folders (`src/q1/outputs`,
`src/q2/outputs`, and so on).

Likewise, you will find the pre-computed outputs in `src/q1/_outputs` (and so on). The figures in the `_outputs` folders
were used in the report. The figures in `outputs` after the code is run should all look lke the figures in the `_outputs`.

### Compute time
These compute times are based on running the code within the Docker image on a 2021 M1 MacBook Pro, 16GB RAM.
- Q1 ~ 2 seconds
- Q2 ~ 1 second
- Q3 ~ 11 seconds
- Q4 ~ 3 minutes
- Q5 ~ 5 minutes

**Note:** Q4 an Q5 make use of `multiprocessing` to speed up the computation. The scripts will attempt to use 8 cores
by default, but if your machine has less than 8 cores, it will use the number of cores available on your machine minus 2,
so that your machine is still usable while the code is running.

### Removing the Container and Image

You can remove the container and image after you are done with them by running
```bash
make clean  # this will remove the container
make clean-image  # this will remove the image
```

## <a name="report"></a> 2. Report
The report is located at [report/out/main.pdf](report/out/main.pdf).
