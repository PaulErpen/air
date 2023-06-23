# Team information

Student 1 ID + Name: 

Student 2 ID + Name: 12107369, Erpenstein Paul

Student 3 ID + Name:



# Report

## Introduction

In this exercise we were tasked with solving three different information retrieval related tasks.
We decided that each team-member should be responsible for their own task.
Our rational behind that was that for each task there would be one person that is the expert on the topic that is treated by that task.
We updated each other on the progress and lessons learned in multiple group meetings over the duration of the exercise.

We divided the work in the following way:

* Part 1: Julia Putz
* Part 2: Paul Erpenstein
* Part 3: Dario Giovanni

## Part 1: Data aggregation process

## Part 2: Neural Re-Ranking

In the second part of the exercise we had to implement, 
two different type of neural reranking models, train them
and finally test them on different data sets.

This was my first time working with a fully-fledged deep-learning library.
I decided to first learn PyTorch by following some of the tutorials that were provided on the PyTorch website.
Afterwards I started with an exploration of the provided code.
This is where I encountered the first major problem, that I had to solve.

### Challenge 1: Version issues

**The problem:** The packages specified in the `requirements.txt` file require a python of version `3.6`.

```{sh}
$ pip install -r requirements.txt 
...
ERROR: Ignored the following versions that require a different python version: 0.2.0 Requires-Python ==3.6
...
ERROR: No matching distribution found for torch==1.6.0
```

This is where I had the first decision to make.
Either I could: 
 * update the packages provided in the `requirements.txt` file to newer versions + adjust the provided code accordingly.
 * create a virtual environment using `conda`.

I did an initial investigation into rewriting the code.
I had to realize that this would be extremely difficult due to multiple reasons:
* Rewriting unfamiliar code in a framework I barely know &rarr; recipe for bugs and misunderstandings
* drastic API changes in the allennlp and pytorch packages across versions 

**The solution:** 

I decided to create a virtual environment.
This decision had far ranging consequences, 
like reconfiguring my IDE and also in how I would run the training.

### Challenge 2: Stability of implementation

**The problem:** The system needed to train, validate, save and test two different models is quite complicated by itself.
There are many components this system needs to interact with like the `data_loading.py` or `core_metrics.py` modules.
Added to this, the internal workings of the implemented models are quite complicated as well.
Simply rerunning the pipeline over and over again and fixing each bug as they occur would be very error-prone and time-consuming.

**The solution:** The implementation of unit tests for each component of the system.
This allowed me to ensure I understood the output of a component, given a specific input.
This was however extremely time-consuming and makes the pipeline much less readable and concise.
I think an alternative solution to this approach would be to create minimal datasets,
so you can manually test and debug the system in a time-efficient manner.

### Challenge 3: Training the models

**The problem:** The models that need to be trained operate on very large tensors.
Running them on my local machine (an underpowered i3 running at 1.20GHz) was completely infeasible.

PyTorch makes it very straightforward to use NVIDIA-GPUs for time-efficient model training.
I considered two possibilities for getting access to a high power GPU:

1. Renting out a virtual server on a cloud service provider like Google Cloud or AWS
2. Google Colab

Initially I looked into option 1., because Colab does not support `python 3.6` natively anymore. I thought this would be much easier than trying to jerry-rig a Colab notebook to somehow use `python 3.6`.
I could not have been more wrong. 
In total I spent two full work days on trying to set up a virtual machine to run the training.
I tried both AWS and Google Cloud.
In both instances there were restrictions to gain access to GPU-powered machines.
AWS straight up refused any request for my account to gain access to such a machine.
Google Cloud allowed me to create and access a machine initially,
but then denied me access for no apparent reason.
In the end I had to accept I had lost a lot of time and reconsider, solving my problem with Colab.

**The solution:** Through a series of linux commands one can install `conda` in a Colab notebook and set up a virtual environment with it.
Then one can us this environment to execute python script.
This is how I managed to run my `python 3.6` code on Colab.

### Evaluation

TODO

### Lessons learned

This exercise had its ups and downs.
Some parts were very frustrating and challenging.
I made a number of mistakes that cost me a lot of time.
Other parts went very well and were extremely fun.
I learned a number of valuable lessons for tackling these sorts of projects in the future.
The lessons are the following...

**Create a minimum viable subset of your data just to test the pipeline:**
This might be a great alternative or complementary strategy to using unit tests.
I did this partly and wish I had just used it more.

**Configuration objects are really handy:**
In future I will probably use configuration objects for my pipelines again.
This makes it easy to have a bundle of values in a single object and switch these objects out depending on your current goals.
This would mean that you could for example have a configuration for testing in your local environment, testing in the server environment and for running the full pipeline.

**Renting hardware (even in virtual form) from cloud service providers in surprisingly hard:**
I though that provisioning your own virtual machines with attached GPUs would be very straight-forward.
However this was not at all the case.
I suspect this has something to do with my AWS- and Google-Cloud-accounts being completely new.
One might have to earn more "trust" with the cloud service provider to have higher priority on the machines.

**PyTorch is surprisingly straightforward:**
I was initially quite intimidated by Deep Learning frameworks.
Using PyTorch for this project showed me that I do not have to be.
If you understand the theoretical concepts of Deep Learning,
using this framework is actually very intuitive.

## Part 3: Extractive QA