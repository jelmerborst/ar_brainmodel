# Brain model of associative recognition

This repository contains the model code associated with the following paper:

* Borst, J.P., Aubin, S., & Stewart, T.C. (under review). A Whole-Task Brain Model accounting for both Human Behavior and Neuroimaging Data. 

In this paper we present a large-scale brain model (810k neurons) that accounts for behavior and neuroimaging data of an associative recognition task. The model was developed using the [Nengo framework](https://github.com/nengo/nengo) and the [Semantic Pointer Architecture](https://github.com/nengo/nengo/tree/master/nengo/spa).

### Setup
* Install [conda](https://docs.conda.io/projects/conda/en/latest/).
* Create a conda environment using `conda_env_brainmodel.yml`: `conda env create --file conda_env_brainmodel.yml` (see for example [this page](https://kiwidamien.github.io/save-the-environment-with-conda-and-how-to-let-others-run-your-programs.html))

### Running the model
There are two options to run the model:
* Run [nengo](https://www.nengo.ai/nengo/examples.html#introductory-tutorials) and open `AR16_full_subj1.py`. Press the play button to run a single trial in the GUI.
* Call `python AR16_full_subj1.py` to run a full experiment (this will take a significant amount of time)

To train the declarative and familiarity memory systems `train_declarative_memory.py` and `train_familiarity_memory.py` were used.
