# learning-joint-limits
Code for the experiments in "Probabilistic Inference of Human Capabilities from Passive Observations", submitted to IROS 2024.
 
 ## Setting Up the Environment
 To set up the virtual Python environment and then run it, run the following commands:
 ```bash
python3.8 -m venv learning_limits
source learning_limits/bin/activate

```

Once in the environment, install the necessary dependencies using the requirements file provided:
```bash
pip install -r requirements.txt
```

You should now be able to run the experiments.

## Running the Experiments

### 3D Manipulator

In order to run the 3D manipulator code, execute the following commands while in the virtual environment:

```bash
cd 3D_Manipulator/
python3 Manipulator_3D.py

```

There are several flags that allow changes to parameters of the script:
- `--verbose`: Enable verbose mode for detailed logging. Default: false.
- `--samples`: Number of samples for the individual task. Default: 2000.
- `--thresh`: Threshold for accepting a sample. Default: 0.02.
- `--cartesian_thresh`: Threshold for accepting a sample with Task space trajectories. Default: 0.01.
- `--kernel`: KDE Kernel width parameter. Default: 0.5
- `--runs`: Number of experiment runs with the same goals. Default: 30.
- `--goals`: Number of different goals (tasks). Default: 10.

The results will be output in a pickle file. To analyse them, run the `analysis.ipynb` notebook.


### HRI Scenario

To run the HRI scenario code, execute the following commands while in the virtual environment:

```bash
cd HRI_Scenario/
python3 Main.py

```
There are several flags that allow changes to parameters of the script:
- `--samples`: Number of samples for the individual task. Default: 5000.
- `--thresh`: Threshold for accepting a sample. Default: 1.5.
- `--cartesian_thresh`: Threshold for accepting a sample with Task space trajectories. Default: 2.5.
- `--kernel`: KDE Kernel width parameter. Default: 0.5
- `--runs`: Number of experiment runs with the same goals. Default: 1.
- `--observations`: Number of different observations provided to the learner. Default: 10.

The results will be output to a results folder. To analyse them, run the `plotting.ipynb` notebook. 

