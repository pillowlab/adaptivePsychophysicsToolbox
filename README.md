# adaptivePsychophysicsToolbox

Matlab package for adaptive psychophysics experiment and analysis. Enables efficient inference of psychometric functions with multiple-alternative responses and lapses. Multinomial logistic response model is used to describe the psychometric function.

A key feature of the package is Bayesian adaptive stimulus selection.
On each trial, the program takes a newly "observed" stimulus-response pair, infers the posterior distribution over the model parameters, and selects a stimulus for the next trial to maximize the expected information gain (infomax).

### Reference:

- Bak JH & Pillow JW (2018). Adaptive stimulus selection for multi-alternative psychometric functions with lapses. [(bioRxiv)](https://doi.org/10.1101/260976)



## Download

* **Download**:   zipped archive  [adaptivePsychophysicsToolbox-master.zip](https://github.com/pillowlab/adaptivePsychophysicsToolbox/archive/master.zip)
* **Clone**: clone the repository from github: ```git clone https://github.com/pillowlab/adaptivePsychophysicsToolbox.git```



## Example scripts

There are three demo scripts, illustrating three possible uses of the package. 
To run, launch matlab and make sure to first cd into the directory containing the code
 (e.g. `cd [yourpath]/adaptivePsychophysicsToolbox/`).
 
In the demo scripts for adaptive stimulus selection, the response is either simulated from a model or drawn from an existing dataset. In practice, this program is supposed to run in a "closed loop" with the actual data collection process, adaptively selecting the next stimulus on each trial.

<!--The tags `_1D` or `_2D` in the demo file names indicate the dimensionality of the stimulus space (a psychometric function on a d-dimensional stimulus space is often called a "d-dimensional psychometric function"). We used binary responses in the 1D case (equivalent to the familiar 2AFC task), and 4-alternative responses in the 2D case. The package can be easily applied to other combinations of stimulus/response dimensions.-->


### Psychometric function inference

```demo1_inferPFs_1D.m``` - infers a psychometric function from a stimulus-response dataset. The demo script reproduces Figure 2 in the paper.


### Simulated experiment

```demo2_AdaptiveStimSelect_2D.m``` - simulates an experiment where stimulus is adaptively selected for each next trial, and response is drawn from a (hidden) fixed model given a stimulus.
This reproduces the simulated experiment shown in Figures 5-6 in the paper.


### Dataset re-ordering

```demo3_reorderingExpt_2D.m``` - illustrates how the package can be used to obtain an optimal re-ordering of an existing dataset.
This demo reproduces the dataset re-ordering analysis shown in Figure 8 in the paper.
The dataset was generated from a (hidden) model in this case, but can be easily replaced by a user data with some simple formatting.



