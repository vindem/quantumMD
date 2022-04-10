# Impact of Hyperparameter Selection on VQE results: A Preliminary Study

This repo contains code and notebooks about Quantum Computing experiments.

[![DOI](https://zenodo.org/badge/459065547.svg)](https://zenodo.org/badge/latestdoi/459065547)

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
