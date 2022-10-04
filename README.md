# Sub-Task Decomposition Enables Learning in Sequence to Sequence Tasks 
Official implementation of the experiments in  the "Sub-Task Decomposition Enables Learning in Sequence to Sequence Tasks" paper.


The following example illustrates how to run a single experiment:

```bash
export steps=true
export depth=12
export width=768
export num_heads=12
export training_iterations=1000000
export warmup_steps=1000
export learning_rate=1e-6
export weight_decay=0
export num_of_bits=128
export seed=27
export greedy_decoding=true
export additional_args=""
export PL_FAULT_TOLERANT_TRAINING=1

./bit_subset_parity_training.sh
```

If you use this codebase, we would appreciate if you cite us as follows:
```bibtex
@article{wies2022sub,
  title={Sub-Task Decomposition Enables Learning in Sequence to Sequence Tasks},
  author={Wies, Noam and Levine, Yoav and Shashua, Amnon},
  journal={arXiv preprint arXiv:2204.02892},
  year={2022}
}
```
