# MHABR.jl

[![CI](https://github.com/ChenxuanyinZou/MHABR.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ChenxuanyinZou/MHABR.jl/actions/workflows/CI.yml)

MHABR.jl implements the **Moving-Horizon Approximate Branch-and-Reduce
(MHABR)** method for learning near-optimal classification trees. It returns an
inspectable recursive tree while retaining the legacy parameters used by the
paper experiments.

## Installation

MHABR.jl requires Julia 1.11. Install the current GitHub version with:

```julia
using Pkg
Pkg.add(url="https://github.com/ChenxuanyinZou/MHABR.jl")
```

## Quick Start

The bundled Iris dataset provides a deterministic three-class example:

```julia
using MHABR
using Random

X, y = load_iris()
train = vcat(1:40, 51:90, 101:140)
test = vcat(41:50, 91:100, 141:150)

tree = fit(
    X[train, :],
    y[train];
    max_depth=2,
    epsilon=0.0,
    alpha=0.0,
    batch=1.0,
    rng=MersenneTwister(1),
)

predictions = predict(tree, X[test, :])
accuracy = score(tree, X[test, :], y[test])
print_tree(tree)
```

The fixed stratified split uses 120 training and 30 test observations and
produces `accuracy == 1.0` with the configuration above. This example
demonstrates the API and is not a paper benchmark result. Run the complete
example with:

```bash
julia --project=. examples/iris.jl
```

See [`examples/data/README.md`](examples/data/README.md) for Iris attribution.

## Tree API

`fit` and `mhabr` return an `MHABRTree` containing recursive `SplitNode` and
`LeafNode` objects. Split nodes store a feature, threshold, and children;
leaves store a prediction, sample count, and class counts. Labels may be
integers, strings, or symbols, and features must be finite real values without
`missing`.

Retrieve the legacy array representation with:

```julia
A, B, C = parameters(tree)
```

The historical tuple-returning interface remains available:

```julia
A, B, C, loss = mh(X_float64, y_integer, depth, epsilon, alpha, batch)
```

The legacy `mh` interface requires `Float64` features and integer labels. New
code should use `fit` or `mhabr`.

## Paper Code

The historical experiment runner is available at
[`scripts/run_experiments.jl`](scripts/run_experiments.jl). It expects
author-prepared numeric train, validation, and test splits under `data/`.

Benchmark datasets and the raw-data preprocessing pipeline are not included
in this package. Dataset indices used by the runner are listed in
[`Dataset_information.xlsx`](Dataset_information.xlsx).

## Reproducibility Warning

- Pass a seeded RNG through `rng`, especially when `batch < 1`.
- The experiment runner derives a deterministic seed from dataset, run, and
  alpha-grid index.

> **Warning:** For historical reproduction, `flag != 0` selects `alpha` using
> test accuracy. Do not use this mode for a new unbiased study. Select on
> validation data, retrain on training plus validation, and evaluate test once.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests cover the legacy algorithm, recursive API, bundled Iris data,
deterministic sampling, soft time limits, input validation, and experiment-run
indexing.

## License and Citation

MHABR.jl is released under the [MIT License](LICENSE). See
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for DecisionTree.jl attribution
and [`CITATION.cff`](CITATION.cff) for citation metadata.
