# JPQ Replicability Study - Virtual Appendix

This is the virtual appendix for the JPQ Replicability paper currently under review.

The folders are as follows:
 - ablation_amd_3090: This contains the JPQ runs for the ablations in Table 2 on the TCT model. As stated in the paper, for this folder, we use only AMD Ryzen CPU with Nvidia 3090 for this experiment.
 - MBitsscan: This contains the JPQ runs for the heatmap of Figure 2, on the TCT model. As stated in the paper, for this folder, this doesnt use IBNs, JPQ negs nor LambdaRank for efficiency reasons.
 - basesetting: This contains the JPQ runs for the various models in Table 1 & Table 3.
 - lion & repllama: This contains the JPQ runs for the various models in Table 4
 - sparse_retrieval: This contains the runs for the SPLADE and BM25 for Table 5.
 - RAG: This contains everything for the NQ experiments in Table 6.
 - indices: This contains Python files to create the Flat indices, using pyterrier_dr and ir_datasets. Flat indices are needed for before JPQ training commences.

Each folder contains: 
 - run files for each variant of JPQ, on the following query sets dev.small, TREC-2019 and TREC-2020.
 - the log files for the training run
 - a listing of the files for the checkpoints and index data structures (these are not provided in the repository, for reasons of space); we provide ONE example, discussed below

The files in the root folder are:
 - preruns.ipynb: a notebook that constructs the invocations of pyterrier_jpq/jpq/run.py to run the various JPQ training runs.
 - tables.ipynb: code to generate the tables in the paper from the log and run files.
 - record_files.py: code to generate file listing for the checkpoint folders.

To install and run the JPQ training code, see the [pyterrier_dr repository](https://github.com/terrierteam/pyterrier_dr)

Want to try it out? See our notebook for MSMARCO: [pyterrier_dr/examples/jpq-tct-replication.ipynb](https://github.com/terrierteam/pyterrier_dr/blob/main/examples/jpq-tct-replication.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrier-org/pyterrier/blob/master/examples/notebooks/indexing.ipynb)

## Citation

A Replicability Study of Joint Product Quantisation for Effective Space-Efficient Dense Retrieval. Craig Macdonald, Nicola Tonellotto, Zhili Shen. In Proceedings of ACM SIGIR 2026. 
