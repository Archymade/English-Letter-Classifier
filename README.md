# Project Title:
### English Alphabet Classification
___

#### Overview
This project was carried out with the aim to predict the identity of handwritten English alphabets based on information encoded via image pixels. The data was sourced from the [OpenML website](www.openml.org).


#### Motivation


#### Quick Start
The datasets for this project for this project may be found under the `train`, `test`, and `valid` subdirectories under the `data/dataset` directory.

To run the scripts, type as below in the Terminal:

1. Navigate to the `scripts` directory.
```
./ $ cd scripts
```
2. Next, run the `main.py` file with the following syntax:

    `py main --argument argument_value`

Example:

```
./scripts/ $ py main.py --n_jobs -1
```
Acceptable arguments include:
- visualize (default = False)
- r_state (default = 42; random state)
- data_dir (data directory)
- train (create train split?)
- valid (create valid split?)
- test (create test split?)

Others may be found in the `main.py` script.

3. Generated diagnostics, text and images, will populate the `reports/text` and `reports/images` directories respectively.
4. Find trained model artefact in the `artefacts` directory.


#### Performance Report
The learning algorithm selected for the project was a support vector machine (SVM). After training, a test set performance of `~ 92 %` was recorded across board for the major classification metrics (accuracy, f1-score, recall, and precision), via `macro` averaging.



#### To-Dos

#### Citation(s)

