# adaptivePsychophysicsToolbox

Package for adaptive psychophysics experiment and analysis. Enables efficient inference of psychometric functions with multiple-alternative responses and lapses. Multinomial logistic response model is used to describe the psychometric function.

This is a Matlab implementation of the algorithms presented in [Bak & Pillow (2018)](https://www.biorxiv.org/content/early/2018/02/06/260976).

## Adaptive stimulus selection

A key feature of the package is Bayesian adaptive stimulus selection.
On each trial, the program takes a newly "observed" stimulus-response pair, infers the posterior distribution over the model parameters, and selects a stimulus for the next trial to maximize the expected information gain (infomax).

In the demo scripts for adaptive stimulus selection, the response is either simulated from a model or drawn from an existing dataset. In practice, this program is supposed to run in a "closed loop" with the actual data collection process, adaptively selecting the next stimulus on each trial.



## Example scripts

There are three demo scripts, illustrating three possible uses of the package. 

The tags `_1D` or `_2D` in the demo file names indicate the dimensionality of the stimulus space (a psychometric function on a d-dimensional stimulus space is often called a "d-dimensional psychometric function"). We used binary responses in the 1D case (equivalent to the familiar 2AFC task), and 4-alternative responses in the 2D case. Of course, the package can be easily applied to other combinations of stimulus/response dimensions.


### Psychometric function inference

```demo1_inferPFs_1D.m```

The simplest use of this package is to fit a psychometric function from a stimulus-response dataset. The demo script reproduces Figure 2 in the paper.


### Simulated experiment

```demo2_AdaptiveStimSelect_2D.m```

This demo script simulates an experiment where stimulus is adaptively selected for each next trial, and reproduces the results in Figures 5-6 in the paper. On each trial in a simulated experiment, response is drawn from a (hidden) fixed model given a stimulus.



### Dataset re-ordering

```demo3_reorderingExpt_2D.m```

This demo script illustrates how the package can be used to obtain an optimal re-ordering of an existing dataset, and reproduces the results in Figure 8 in the paper.
The dataset was generated from a (hidden) model in this case, but can be easily replaced by a user data with some simple formatting.


<!--On each trial in a re-ordering analysis, the program draws a stimulus-response pairs from a large existing dataset D<sub>0</sub>, and pretend that it is a new "observation" from the trial (add it to a reordered dataset D). 
The reordered dataset D is then considered the "new" experimental result; the base dataset D<sub>0</sub> is not directly used in the inference nor in the adaptive stimulus selection algorithm.-->

<!--Once the next stimulus x* is selected by the algorithm, the next trial draws randomly from all stimulus-response _pairs_ with the selected stimulus x*.-->


## More documentation

See the documentation in the main program. You can see this by doing:

- make sure you are in the root directory of the package;
- run `setpaths` in the command line of Matlab;
- run `help prog_infomax_MNLwL`.

## Reference

- Bak JH & Pillow JW (2018). Adaptive stimulus selection for multi-alternative psychometric functions with lapses. [(bioRxiv)](https://www.biorxiv.org/content/early/2018/02/06/260976)