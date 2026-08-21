# MHABR.jl

MHABR.jl implements the **Moving-Horizon Approximate Branch-and-Reduce
(MHABR)** method for training near-optimal deep classification trees within a
bilevel optimization framework.

The package returns a recursive, inspectable tree. It also preserves the
legacy array parameters used by the paper experiments so that existing results
can be reproduced.

## Installation

MHABR.jl supports Julia 1.11. Its direct dependencies and compatibility bounds
are declared in [`Project.toml`](Project.toml).

From a local checkout, instantiate the environment and run the tests with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

To develop the package from this checkout in another Julia environment, start
Julia in the repository directory and run:

```julia
using Pkg
Pkg.develop(path=pwd())
using MHABR
```

After the repository is published, its HTTPS clone address can be passed to
`Pkg.add(url=...)`. After registration in the Julia General Registry, install
the package with:

```julia
using Pkg
Pkg.add("MHABR")
```

## Quick Start

```julia
using MHABR
using Random

X = [0.0; 1.0;;]
y = ["left", "right"]

tree = fit(
    X,
    y;
    max_depth=1,
    epsilon=0.0,
    alpha=0.0,
    batch=1.0,
    rng=MersenneTwister(1),
)

predict(tree, X)
predict_proba(tree, X)
score(tree, X, y)
print_tree(tree)
```

`fit` and the algorithm-specific `mhabr` function return an `MHABRTree`.
Integer, string, and symbol class labels are supported. Feature values must be
finite real numbers without `missing` values.

## Tree Representation

An `MHABRTree` contains recursive `SplitNode` and `LeafNode` objects:

- A `SplitNode` stores a feature index, a threshold, and its left and right
  children.
- A `LeafNode` stores its predicted label, training-sample count, and class
  counts.
- Branches that are inactive in the full binary-tree encoding are folded into
  leaves in the public representation.

The exact legacy array representation associated with the fitted tree remains
available:

```julia
A, B, C = parameters(tree)
```

For existing experiment code, `mh` retains the tuple-returning interface:

```julia
A, B, C, loss = mh(X_float64, y_integer, depth, epsilon, alpha, batch)
```

The legacy `mh` interface requires `Float64` features and integer labels. New
applications should use `fit` or `mhabr`.

## Reproducing the Paper Experiments

Run the historical benchmark runner from the repository root:

```bash
julia --project=. scripts/run_experiments.jl \
  <ds_start> <ds_end> <run_start> <run_end> \
  <time_limit> <tree_depth> <epsilon> <batch> <flag>
```

For example, the following command runs the 51 small datasets over ten
independent partitions using depth-four trees, exact split search, and no
mini-batch sampling:

```bash
julia --project=. scripts/run_experiments.jl 1 51 1 10 10 4 0 1 0
```

### Command-Line Arguments

| Argument | Type | Description |
|:--|:--|:--|
| `ds_start` | `Int` | First dataset index to process, inclusive. |
| `ds_end` | `Int` | Last dataset index to process, inclusive. |
| `run_start` | `Int` | First preprocessed data partition to run, inclusive. |
| `run_end` | `Int` | Last preprocessed data partition to run, inclusive. |
| `time_limit` | `Float64` | Soft MHABR time limit in seconds for each fitted tree. |
| `tree_depth` | `Int` | Maximum tree depth; must be at least 1. |
| `epsilon` | `Float64` | Nonnegative approximate-search termination tolerance. `0` requests the full split search. |
| `batch` | `Float64` | Sampling fraction used to evaluate candidate subtrees; must lie in `(0, 1]`. Use `1` for all observations. |
| `flag` | numeric | Use `0` for `alpha = 0`; a nonzero value enables the historical alpha-tuning grid. |

`time_limit` is a soft limit. MHABR checks the deadline between moving-horizon
nodes, features, and recursive split evaluations. A CART fit that has already
started may finish before control returns to the deadline check.

## Dataset Organization

The prepared reproducibility datasets contain ten independent partitions per
dataset. Each partition uses:

- 50% of observations for training;
- 25% for validation;
- 25% for testing.

The historical default runner combines the training and validation partitions
before fitting and reserves the test partition for evaluation. Dataset-index
to dataset-name mappings are provided in
[`Dataset_information.xlsx`](Dataset_information.xlsx).

| Index range | Category | Paper reference |
|:--|:--|:--|
| 1–51 | Small-scale datasets (`n < 10,000`) | Table 8 and Section 5.1 |
| 52–56 | Medium-scale datasets (`10,000 ≤ n ≤ 1,000,000`) | Table 2 |
| Not indexed | Large-scale datasets: SUSY, HIGGS, and WESAD | Scalability experiments |

The large datasets are not bundled with the package repository. SUSY and
HIGGS are available through the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/); WESAD should
be obtained from its original public repository. The complete preprocessed
paper archive should be published separately through a versioned GitHub
Release or Zenodo record.

## Reproducibility Notes

- Pass a seeded random-number generator through `rng` when calling the package
  API, especially when `batch < 1`.
- The experiment runner derives a deterministic seed from the dataset, run,
  and alpha-grid index.
- `parameters(tree)` returns copies; modifying them does not mutate the fitted
  tree.
- `Manifest.toml` records the dependency versions used for the archived paper
  environment, while `Project.toml` defines the package's supported dependency
  ranges.

### Historical Alpha-Tuning Behavior

When `flag != 0`, the runner intentionally preserves the anonymous paper-code
behavior of selecting `alpha` using test accuracy and emits a warning. This
mode exists only to reproduce historical results. It must not be used for a
new unbiased evaluation.

For a new study, fit the candidate models on the training partition, select
`alpha` on validation data, retrain once on training plus validation data, and
evaluate the test partition once.

## Testing

Run the full package test suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests cover the legacy array algorithm, recursive tree API, deterministic
sampling, soft time limits, input validation, and experiment-run indexing.

## License and Attribution

MHABR.jl is released under the [MIT License](LICENSE). The bundled
classification-tree implementation is modified from DecisionTree.jl. See
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for upstream copyright and
license information.
