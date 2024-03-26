
# RankingSHAP

## Description
This repository contains the implementation of RankingSHAP, discussed in our paper titled "RankingSHAP – Listwise Feature Attribution Explanations
for Ranking Models". With this code we demonstrate the ability of RankingSHAP to generate feature attribution explanations for ranking models. We
compare to several baselines, showing that the only approach that can accurately approximate feature attribution, as defined by marginal
contributions and shapley values for the ranking application is RankingSHAP.

## Getting Started


### Dependencies
A list of dependencies can be found in the requirements.txt file. The code is tested on python 3.11.

### Installing
First clone this repository to your local machine:

```
git clone git@github.com:MariaHeuss/RankingShap.git
```

Navigate to the cloned repository directory:

```
cd RankingShap
```
Create a virtual environment on your machine and install the dependencies
```
pip install -r requirements.txt
```

### Collecting Data
Before running the scripts, ensure you have collected all the necessary data. For our paper we have tested two datasets
from LETOR4.0. MQ2008 and MSLR-WEB10K. You can download those datasets by following these steps:

- Create a folder data
- Download the dataset from `https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/`.
Follow the provided link to OneDrive and download the file named `MQ2008.rar`.
- Unpack the data and place the dataset in a folder called MQ2008 in the data directory within the project.
- The folder structure should be for example `data/MQ2008/Fold1/test.txt`.

In the same way download the MSLR-WEB10K data from `https://www.microsoft.com/en-us/research/project/mslr/` and store it
in a folder called MSLR-WEB10K in the data folder.


### Executing Program
To run the main scripts of the project, execute the following commands in the terminal:

```
run_rankingshap.bash
```

This script will execute a number of scripts that sequentially train a model, generate explanations,
generate ground truth attribution labels and evaluate the explanations with those for the MQ2008 data. You can test the
code first by running

```
run_rankingshap_test.bash
```
which executes the same code but for only one query.

## Citation
If you use this code or our results in your research, please cite our paper as follows:

```
@article{heuss2024,
  title={RankingSHAP – Listwise Feature Attribution Explanations
for Ranking Models},
  author={Heuss,Maria and deRijke, Maarten and Anand, Avishek},
  year={2024}
}
```

## License
This repository is published under the terms of the GNU General Public License version 3. For more information, see the file LICENSE.
```
RankingSHAP – Listwise Feature Attribution Explanations
for Ranking Models
Copyright (C) 2024 Maria Heuss

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
```

## Contact
For any queries, please reach out to m dot c dot heuss at uva dot nl.
