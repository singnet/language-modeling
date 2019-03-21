# Bert Classifier Visualization BERTRAM

## Description
This repository includes scripts that visualize information on the classifier layer.
Used the pretrained BERT model from pip or model trained on Gutenberg Children dataset.

## Installation

* Create anaconda enviroment with python 3.6:

`$ conda create -n py36 python=3.6 anaconda`

* Activate enviroment:

`$ conda activate py36`

* Install dependencies:

`$ conda install -c conda-forge websockets`

`$ conda install -c plotly plotly`

`$ pip install pytorch-pretrained-bert`

## Launch 

* Activate enviroment:

`$ conda activate py36`

* Launch without mask:

`$ python mi_base.py`

Sentence example without mask:

*After a time she heard a little pattering of feet in the distance, and she hastily dried her eyes to see what was coming.*

* Launch with mask without ground truth (`[MASK] -> predicted token`):

`$ python mi_mask_prediction.py`

Sentence example with mask prediction:

*After a time she heard [MASK] little pattering of feet in the distance, and she hastily dried her eyes to see what was coming.*

* Launch with mask with ground truth (`[MASK] -> ground truth token`):

`$ python mi_mask_ground_truth.py`

Sentence example with mask ground truth:

*After a time she heard {{a}} little pattering of feet in the distance, and she hastily dried her eyes to see what was coming.*

* Launch with multi mask ground truth (`every token as mask ground truth in its iteration`):

`$ python mi_multi_mask_ground_truth.py`

Sentence example with multi mask ground truth in its iteration:

*After a time she heard a little pattering of feet in the distance, and she hastily dried her eyes to see what was coming.*

For change sentence as default. Change variable `sentence` in needed script. 

## Note 

This script can work as a web-socket server. This part of the script is commented out for local launch.