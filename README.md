# Impact of Hyperparameter Selection on VQE results: A Preliminary Study

This repo contains code and notebooks about Quantum Computing experiments.

[![DOI](https://zenodo.org/badge/459065547.svg)](https://zenodo.org/badge/latestdoi/459065547)

Jupyter Notebooks:
* Quantum Molecular Dynamics Experiments: quantumMD
* Comparison between VQAs using different PQCs: quantum-pqc-comparison

Results:
runtume_tests_real.csv: Comparison of two different VQE methods (using different PQCs) on a real backend (runtime and Normalized Root Mean Squared Error,                           compared to the classical implementation of eigensolver)
runtune_tests_sim.csv: Comparison of two different VQE methods (using different PQCs) on a simulated backend (runtime and Normalized Root Mean Squared                            Error, compared to the classical implementation of eigensolver)

Note: Simulator assumes full qubit connectivity, therefore performance is always better than real backend for low number of qubits.
