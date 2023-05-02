Please cite this work using the following bibtex entry:

@inproceedings{DBLP:conf/eScience/CranganoreMBDD22,
  author       = {Sandeep Suresh Cranganore and
                  Vincenzo De Maio and
                  Ivona Brandic and
                  Tu Mai Anh Do and
                  Ewa Deelman},
  title        = {Molecular Dynamics Workflow Decomposition for Hybrid Classic/Quantum
                  Systems},
  booktitle    = {18th {IEEE} International Conference on e-Science, e-Science 2022,
                  Salt Lake City, UT, USA, October 11-14, 2022},
  pages        = {346--356},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/eScience55777.2022.00048},
  doi          = {10.1109/eScience55777.2022.00048},
  timestamp    = {Tue, 21 Mar 2023 20:57:08 +0100},
  biburl       = {https://dblp.org/rec/conf/eScience/CranganoreMBDD22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


# Impact of Hyperparameter Selection on VQE

This repo contains code and notebooks about Quantum Computing experiments.

[![DOI](https://zenodo.org/badge/459065547.svg)](https://zenodo.org/badge/latestdoi/459065547)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8B-orange)](https://fair-software.eu)

## Acknowledgements ##
We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the authors, and do not reflect the official policy or position of IBM or the IBM Quantum team. [IBM Quantum](https://quantum-computing.ibm.com/) 

Machines used in this work:
| ID | VERSION | PROCESSOR |
|----|---------|-----------|
| ibmq_manila |1.0.29|Falcon r5.11L|
| ibmq_santiago|1.4.1|Falcon r4L|

## Jupyter Notebooks: ##
* quantumMD 
A preliminary study of applying quantum computing to scientific computation 
* quantum-pqc-comparison
An example of performance of different PQC for variational algorithms

## Data ##
Filename format: ALGORITHM_BACKEND_QUBITS_[RT-NRMSE|SUMMARY]
* rt-nrmse: summary of the experiment with average runtime and normalized root mean square error between the value obtained on classic architecture and the value obtained with specific PQC;

| PQC | AVG-RUNTIME | NRMSE |
|-----|-------------|-------|
| PQC name | Average RT for circuit | NRMSE for circuit |

* rawdata: data of each execution of VQE over which rt-nrmse are calculated.

| PQC0-RT | PQC0-EIG | ... | PQCn-RT | PQCn-EIG | Classic-EIG |
|-------|--------|---|-------|--------|-----------|
| RT matrix #1 using PQC0 | EIG matrix #1 using PQC0 |...| RT matrix #1 using PQCn | EIG matrix #1 using PQCn | EIG matrix #1 classic |
|.......|........|...|.......|........|...........|
| RT matrix #m using PQC0 | EIG matrix #m using PQC0 |...| RT matrix #m using PQCn | EIG matrix #m using PQCn | EIG matrix #m classic |



**Note: Simulator assumes full qubit connectivity, therefore performance is always better than real backend for low number of qubits.**
