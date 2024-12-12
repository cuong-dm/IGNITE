# IGNITE: Incorporating Surrogate Gradient Norm to Improve Offline Optimization Techniques

## Requirements

To install environment:

Following instructions from https://github.com/kaist-silab/design-baselines-fixes/tree/main.


## Training

To run the vanilla baselines in the paper, run this command:

```train
bash scripts/baseline.sh
```

To run the baselines w/ IGNITE in the paper, run this command:

```train
bash scripts/ignite.sh
```

To run the baselines w/ IGNITE2 in the paper, run this command:

```train
bash scripts/ignite2.sh
```

## Evaluation

To evaluate 100-th percentile performance, run:

```eval
design-baselines make-table --dir <result_folder> --percentile 100th
```

To evaluate 80-th percentile performance, run:

```eval
design-baselines make-table --dir <result_folder> --percentile 80th
```

To evaluate 50-th percentile performance, run:

```eval
design-baselines make-table --dir <result_folder> --percentile 50th
```
