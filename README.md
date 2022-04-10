# Impact of Hyperparameter Selection on VQE results: A Preliminary Study

This repo contains code and notebooks about Quantum Computing experiments.

[![DOI](https://zenodo.org/badge/459065547.svg)](https://zenodo.org/badge/latestdoi/459065547)

## Jupyter Notebooks: ##
* quantumMD 
A preliminary study of applying quantum computing to scientific computation 
* quantum-pqc-comparison
An example of performance of different PQC for variational algorithms

## Results ##
* runtime_tests_real.csv
*Comparison of two different VQE methods (using different PQCs) on a real backend (runtime and Normalized Root Mean Squared Error, compared to the classical implementation of eigensolver)
* runtime_tests_sim.csv
*Comparison of two different VQE methods (using different PQCs) on a simulated backend (runtime and Normalized Root Mean Squared Error, compared to the classical implementation of eigensolver)

**Note: Simulator assumes full qubit connectivity, therefore performance is always better than real backend for low number of qubits.**
