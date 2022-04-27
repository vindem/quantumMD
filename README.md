# Impact of Hyperparameter Selection on VQE

This repo contains code and notebooks about Quantum Computing experiments.

[![DOI](https://zenodo.org/badge/459065547.svg)](https://zenodo.org/badge/latestdoi/459065547)

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
* ibmq_qasm_simulator_rawdata.csv

*Comparison of two different VQE methods (using different PQCs) on a simulated backend, single measurements of eigenvalues and runtime

* ibmq_qasm_simulator_summary.csv

*Comparison of two different VQE methods (using different PQCs) on a simulated backend, average runtime and nrmse of eigenvalue vs classic





**Note: Simulator assumes full qubit connectivity, therefore performance is always better than real backend for low number of qubits.**
