# Modified from DecisionTree.jl classification-tree code.
# See ../THIRD_PARTY_NOTICE.md for upstream copyright and license notices.
module ClassificationTree
using Random
using StatsBase

export build_tree, build_stump, prune_tree, apply_tree, apply_tree_proba

struct Leaf{T}
    majority::T
    values::Vector{T}
end

struct Node{S,T}
    featid::Int
    featval::S
    left::Union{Leaf{T},Node{S,T}}
    right::Union{Leaf{T},Node{S,T}}
    eveloss::Vector{Float64}
    featbound::Vector{Float64}
end

mutable struct NodeMeta{S}
    l::NodeMeta{S}      # right child
    r::NodeMeta{S}      # left child
    label::Int              # most likely label
    feature::Int            # feature used for splitting
    threshold::S                # threshold value
    eveloss::Vector{Float64}               # second best feature
    threbound::Vector{Float64}          # second best threshold
    # opt_threshold::Vector{S}   # optimal threshold
    # opt_purity::Vector{S}      # optimal purity
    is_leaf::Bool
    depth::Int
    region::UnitRange{Int}     # a slice of the samples used to decide the split of the node
    features::Vector{Int}      # a list of features not known to be constant
    split_at::Int              # index of samples
    node_impurity::Float64
    function NodeMeta{S}(
        features::Vector{Int},
        region::UnitRange{Int},
        depth::Int,
        node_impurity::Float64=0.0,
    ) where {S}
        node = new{S}()
        node.depth = depth
        node.region = region
        node.features = features
        node.is_leaf = false
        node.node_impurity = node_impurity
        node
    end
end

struct Tree{S,T}
    root::NodeMeta{S}
    list::Vector{T}
    labels::Vector{Int}
end

const LeafOrNode{S,T} = Union{Leaf{T},Node{S,T}}

struct Root{S,T}
    node::LeafOrNode{S,T}
    n_feat::Int
    featim::Vector{Float64}   # impurity importance
end

struct Ensemble{S,T}
    trees::Vector{LeafOrNode{S,T}}
    n_feat::Int
    featim::Vector{Float64}
end

is_leaf(l::Leaf) = true
is_leaf(n::Node) = false

_zero(::Type{String}) = ""
_zero(x::Any) = zero(x)
function convert(::Type{Node{S,T}}, lf::Leaf{T}) where {S,T}
    Node{S,T}(0, _zero(S), lf, Leaf(_zero(T), [_zero(T)]))
end
function convert(::Type{Root{S,T}}, node::LeafOrNode{S,T}) where {S,T}
    Root{S,T}(node, 0, Float64[])
end
convert(::Type{LeafOrNode{S,T}}, tree::Root{S,T}) where {S,T} = tree.node
promote_rule(::Type{Node{S,T}}, ::Type{Leaf{T}}) where {S,T} = Node{S,T}
promote_rule(::Type{Leaf{T}}, ::Type{Node{S,T}}) where {S,T} = Node{S,T}
promote_rule(::Type{Root{S,T}}, ::Type{Leaf{T}}) where {S,T} = Root{S,T}
promote_rule(::Type{Leaf{T}}, ::Type{Root{S,T}}) where {S,T} = Root{S,T}
promote_rule(::Type{Root{S,T}}, ::Type{Node{S,T}}) where {S,T} = Root{S,T}
promote_rule(::Type{Node{S,T}}, ::Type{Root{S,T}}) where {S,T} = Root{S,T}

const DOC_WHATS_A_TREE =
    "Here `tree` is any `DecisionTree_modified.Root`, `DecisionTree_modified.Node` or " *
    "`DecisionTree_modified.Leaf` instance, as returned, for example, by [`build_tree`](@ref)."
const DOC_WHATS_A_FOREST =
    "Here `forest` is any `DecisionTree_modified.Ensemble` instance, as returned, for " *
    "example, by [`build_forest`](@ref)."
const DOC_ENSEMBLE = "`DecisionTree_modified.Ensemble` objects are returned by, for example, `build_forest`."
const ERR_ENSEMBLE_VCAT = DimensionMismatch(
    "Ensembles that record feature impurity importances cannot be combined when " *
    "they were generated using differing numbers of features. ",
)
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where {T<:Integer} = Random.MersenneTwister(seed)
# --- Utilities ---

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector, votes::AbstractVector, weights=1.0)
    label2ind = label_index(labels)
    counts = zeros(Float64, length(label2ind))
    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
    N = size(X, 1)
    N_cols = length(row_fun(X[1, :])) # gets the number of columns
    out = Array{Float64}(undef, N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(X[i, :])
    end
    return out
end

# --- Tree Structure & Impurity Management ---

function _convert(
    node::NodeMeta{S}, list::AbstractVector{T}, labels::AbstractVector{T}
) where {S,T}
    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S,T}(node.feature, node.threshold, left, right, node.eveloss, node.threbound)
    end
end

function update_using_impurity!(    # update feature importance using impurity
    feature_importance::Vector{Float64}, node::NodeMeta{S}
) where {S}
    if !node.is_leaf
        update_using_impurity!(feature_importance, node.l)
        update_using_impurity!(feature_importance, node.r)
        feature_importance[node.feature] +=
            node.node_impurity - node.l.node_impurity - node.r.node_impurity
    end
    return nothing
end

nsample(leaf::Leaf) = length(leaf.values)
nsample(tree::Node) = nsample(tree.left) + nsample(tree.right)
nsample(tree::Root) = nsample(tree.node)

# Numbers of observations for each unique labels
function votes_distribution(labels)
    unique_labels = unique(labels)
    votes = zeros(Int, length(unique_labels))
    @simd for label in labels
        votes[findfirst(==(label), unique_labels)] += 1
    end
    votes
end

function update_pruned_impurity!(
    tree::LeafOrNode{S,T},
    feature_importance::Vector{Float64},
    ntt::Int,
    loss::Function=entropy,
) where {S,T}
    all_labels = [tree.left.values; tree.right.values]
    nc = votes_distribution(all_labels)
    nt = length(all_labels)
    ncl = votes_distribution(tree.left.values)
    nl = length(tree.left.values)
    ncr = votes_distribution(tree.right.values)
    nr = nt - nl
    feature_importance[tree.featid] -=
        (nt * loss(nc, nt) - nl * loss(ncl, nl) - nr * loss(ncr, nr)) / ntt
end

# --- Core Training Functions ---

function build_stump(
    labels::AbstractVector{T},
    features::AbstractArray{S},
    weights=nothing;
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T}
    rng = mk_rng(rng)::Random.AbstractRNG
    t = fit(;
        X=features,
        Y=labels,
        W=weights,
        loss=zero_one,
        max_features=size(features, 2),
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
        min_purity_increase=0.0,
        rng,
    )

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function build_tree(
    labels::AbstractVector{T},
    features::AbstractArray{S},
    n_subfeatures=0,
    max_depth=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    min_purity_increase=0.0;
    loss=entropy::Function, # Defaulted to classification entropy
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T}
    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = fit(;
        X=features,
        Y=labels,
        W=nothing,
        loss,
        max_features=Int(n_subfeatures),
        max_depth=Int(max_depth),
        min_samples_leaf=Int(min_samples_leaf),
        min_samples_split=Int(min_samples_split),
        min_purity_increase=Float64(min_purity_increase),
        rng,
    )

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function _build_tree(
    tree::Tree{S,T},
    labels::AbstractVector{T},
    n_features,
    n_samples,
    impurity_importance::Bool,
) where {S,T}
    node = _convert(tree.root, tree.list, labels[tree.labels])
    if !impurity_importance
        return Root{S,T}(node, n_features, Float64[])
    else
        fi = zeros(Float64, n_features)
        update_using_impurity!(fi, tree.root)
        return Root{S,T}(node, n_features, fi ./ n_samples)
    end
end

# --- Pruning & Evaluation ---

function prune_tree(
    tree::Union{Root{S,T},LeafOrNode{S,T}},
    purity_thresh=1.0,
    loss::Function=entropy, # Forced to classification loss
) where {S,T}
    if purity_thresh >= 1.0
        return tree
    end
    ntt = nsample(tree)
    function _prune_run_stump(
        tree::LeafOrNode{S,T}, purity_thresh::Real, fi::Vector{Float64}=Float64[]
    ) where {S,T}
        all_labels = [tree.left.values; tree.right.values]
        majority = majority_vote(all_labels)
        matches = findall(all_labels .== majority)
        purity = length(matches) / length(all_labels)
        if purity >= purity_thresh
            if !isempty(fi)
                update_pruned_impurity!(tree, fi, ntt, loss)
            end
            return Leaf{T}(majority, all_labels)
        else
            return tree
        end
    end
    function _prune_run(tree::Root{S,T}, purity_thresh::Real) where {S,T}
        fi = deepcopy(tree.featim) ## recalculate feature importances
        node = _prune_run(tree.node, purity_thresh, fi)
        return Root{S,T}(node, tree.n_feat, fi)
    end
    function _prune_run(
        tree::LeafOrNode{S,T}, purity_thresh::Real, fi::Vector{Float64}=Float64[]
    ) where {S,T}
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            return _prune_run_stump(tree, purity_thresh, fi)
        else
            left = _prune_run(tree.left, purity_thresh, fi)
            right = _prune_run(tree.right, purity_thresh, fi)
            return Node{S,T}(tree.featid, tree.featval, left, right)
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

# --- Inference ---

apply_tree(leaf::Leaf, feature::AbstractVector) = leaf.majority
function apply_tree(tree::Root{S,T}, features::AbstractVector{S}) where {S,T}
    apply_tree(tree.node, features)
end

function apply_tree(tree::Node{S,T}, features::AbstractVector{S}) where {S,T}
    if tree.featid == 0
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::Root{S,T}, features::AbstractMatrix{S}) where {S,T}
    apply_tree(tree.node, features)
end

function apply_tree(tree::LeafOrNode{S,T}, features::AbstractMatrix{S}) where {S,T}
    N = size(features, 1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    return predictions
end

function apply_tree_proba(tree::Root{S,T}, features::AbstractVector{S}, labels) where {S,T}
    apply_tree_proba(tree.node, features, labels)
end

function apply_tree_proba(leaf::Leaf{T}, features::AbstractVector{S}, labels) where {S,T}
    compute_probabilities(labels, leaf.values)
end

function apply_tree_proba(tree::Node{S,T}, features::AbstractVector{S}, labels) where {S,T}
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

function apply_tree_proba(tree::Root{S,T}, features::AbstractMatrix{S}, labels) where {S,T}
    apply_tree_proba(tree.node, features, labels)
end

function apply_tree_proba(
    tree::LeafOrNode{S,T}, features::AbstractMatrix{S}, labels
) where {S,T}
    stack_function_results(row -> apply_tree_proba(tree, row, labels), features)
end



#### util ##########

function assign(Y::AbstractVector{T}, list::AbstractVector{T}) where {T}
    dict = Dict{T,Int}()
    @simd for i in eachindex(list)  #in 1:length(list)
        @inbounds dict[list[i]] = i
    end

    _Y = Array{Int}(undef, length(Y))
    @simd for i in eachindex(Y)
        @inbounds _Y[i] = dict[Y[i]]
    end

    return list, _Y
end

function assign(Y::AbstractVector{T}) where {T}
    set = Set{T}()
    for y in Y
        push!(set, y)
    end
    list = collect(set)
    return assign(Y, list)
end

@inline function zero_one(ns, n)
    return 1.0 - maximum(ns) / n
end

@inline function gini(ns, n)
    s = 0.0
    @simd for k in ns
        s += k * (n - k)
    end
    return s / (n * n)
end

@inline function normal_loss(ns, n)
    s = Float64(n - maximum(ns))
    return s / n
end

# compute table of values i*log(i) for integers in 0 <= i <= maxvalue
# where tables[i+1] = i * log(i)
# (0*log(0) is set to 0 for convenience when computing entropy)
function compute_entropy_terms(maxvalue)
    entropy_terms = zeros(Float64, maxvalue + 1)
    for i in 1:maxvalue
        entropy_terms[i+1] = i * log(i)
    end
    return entropy_terms
end

# returns the entropy of ns/n, ns is an array of integers
# and entropy_terms are precomputed entropy terms
@inline function entropy(ns::AbstractVector{U}, n, entropy_terms) where {U<:Integer}
    s = 0.0
    for k in ns
        s += entropy_terms[k+1]
    end
    return log(n) - s / n
end

@inline function entropy(ns, n)
    s = 0.0
    @simd for k in ns
        if k > 0
            s += k * log(k)
        end
    end
    return log(n) - s / n
end

# adapted from the Julia Base.Sort Library
@inline function partition!(v, w, pivot, region)
    i, j = 1, length(region)
    r_start = region.start - 1
    @inbounds while true
        while w[i] <= pivot
            i += 1
        end
        while w[j] > pivot
            j -= 1
        end
        i >= j && break
        ri = r_start + i
        rj = r_start + j
        v[ri], v[rj] = v[rj], v[ri]
        w[i], w[j] = w[j], w[i]
        i += 1
        j -= 1
    end
    return j
end

# adapted from the Julia Base.Sort Library
function insert_sort!(v, w, lo, hi, offset)
    @inbounds for i in (lo+1):hi
        j = i
        x = v[i]
        y = w[offset+i]
        while j > lo
            if x < v[j-1]
                v[j] = v[j-1]
                w[offset+j] = w[offset+j-1]
                j -= 1
                continue
            end
            break
        end
        v[j] = x
        w[offset+j] = y
    end
    return v
end

@inline function _selectpivot!(v, w, lo, hi, offset)
    @inbounds begin
        mi = (lo + hi) >>> 1

        # sort the values in v[lo], v[mi], v[hi]

        if v[mi] < v[lo]
            v[mi], v[lo] = v[lo], v[mi]
            w[offset+mi], w[offset+lo] = w[offset+lo], w[offset+mi]
        end
        if v[hi] < v[mi]
            if v[hi] < v[lo]
                v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
                w[offset+lo], w[offset+mi], w[offset+hi] = w[offset+hi],
                w[offset+lo],
                w[offset+mi]
            else
                v[hi], v[mi] = v[mi], v[hi]
                w[offset+hi], w[offset+mi] = w[offset+mi], w[offset+hi]
            end
        end

        # move v[mi] to v[lo] and use it as the pivot
        v[lo], v[mi] = v[mi], v[lo]
        w[offset+lo], w[offset+mi] = w[offset+mi], w[offset+lo]
        v_piv = v[lo]
        w_piv = w[offset+lo]
    end

    # return the pivot
    return v_piv, w_piv
end

# adapted from the Julia Base.Sort Library
@inline function _bi_partition!(v, w, lo, hi, offset)
    pivot, w_piv = _selectpivot!(v, w, lo, hi, offset)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1
        j -= 1
        while v[i] < pivot
            i += 1
        end
        while pivot < v[j]
            j -= 1
        end
        i >= j && break
        v[i], v[j] = v[j], v[i]
        w[offset+i], w[offset+j] = w[offset+j], w[offset+i]
    end
    v[j], v[lo] = pivot, v[j]
    w[offset+j], w[offset+lo] = w_piv, w[offset+j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end

# adapted from the Julia Base.Sort Library
# adapted from the Julia Base.Sort Library
# this sorts v[lo:hi] and w[offset+lo, offset+hi]
# simultaneously by the values in v[lo:hi]
const SMALL_THRESHOLD = 20
function q_bi_sort!(v, w, lo, hi, offset)
    @inbounds while lo < hi
        hi - lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi, offset)
        j = _bi_partition!(v, w, lo, hi, offset)
        if j - lo < hi - j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j - 1) && q_bi_sort!(v, w, lo, j - 1, offset)
            lo = j + 1
        else
            j + 1 < hi && q_bi_sort!(v, w, j + 1, hi, offset)
            hi = j - 1
        end
    end
    return v
end

# The code function below is a small port from numpy's library
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree_modified.jl is released under the MIT license.

# ported by Poom Chiarawongse <eight1911@gmail.com>

# this is the code for efficient generation
# of hypergeometric random variables ported from numpy.random
function hypergeometric(good, bad, sample, rng)
    @inline function loggam(x)
        x0 = x
        n = 0
        if (x == 1.0 || x == 2.0)
            return 0.0
        elseif x <= 7.0
            n = Int(floor(7 - x))
            x0 = x + n
        end
        x2 = 1.0 / (x0 * x0)
        xp = 6.2831853071795864769252867665590 # Tau
        gl0 = -1.39243221690590e+00
        gl0 = gl0 * x2 + 1.796443723688307e-01
        gl0 = gl0 * x2 - 2.955065359477124e-02
        gl0 = gl0 * x2 + 6.410256410256410e-03
        gl0 = gl0 * x2 - 1.917526917526918e-03
        gl0 = gl0 * x2 + 8.417508417508418e-04
        gl0 = gl0 * x2 - 5.952380952380952e-04
        gl0 = gl0 * x2 + 7.936507936507937e-04
        gl0 = gl0 * x2 - 2.777777777777778e-03
        gl0 = gl0 * x2 + 8.333333333333333e-02
        gl = gl0 / x0 + 0.5 * log(xp) + (x0 - 0.5) * log(x0) - x0
        if x <= 7.0
            @simd for k in 1:n
                gl -= log(x0 - k)
            end
        end
        return gl
    end

    @inline function hypergeometric_hyp(good, bad, sample)
        d1 = bad + good - sample
        d2 = min(bad, good)

        Y = d2
        K = sample
        while Y > 0
            Y -= floor(UInt, rand(rng) + Y / (d1 + K))
            K -= 1
            if K == 0
                break
            end
        end

        Z = d2 - Y
        return if good > bad
            sample - Z
        else
            Z
        end
    end

    @inline function hypergeometric_hrua(good, bad, sample)
        mingoodbad = min(good, bad)
        maxgoodbad = max(good, bad)
        popsize = good + bad
        m = min(sample, popsize - sample)
        d4 = mingoodbad / popsize
        d5 = 1.0 - d4
        d6 = m * d4 + 0.5
        d7 = sqrt((popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5)
        # d8 = 2*sqrt(2/e) * d7 + (3 - 2*sqrt(3/e))
        d8 = 1.7155277699214135 * d7 + 0.8989161620588988
        d9 = floor(UInt, (m + 1) * (mingoodbad + 1) / (popsize + 2))
        d10 = (
            loggam(d9 + 1) +
            loggam(mingoodbad - d9 + 1) +
            loggam(m - d9 + 1) +
            loggam(maxgoodbad - m + d9 + 1)
        )
        d11 = min(m + 1, mingoodbad + 1, floor(UInt, d6 + 16 * d7))

        while true
            X = rand(rng)
            Y = rand(rng)
            W = d6 + d8 * (Y - 0.5) / X

            (W < 0.0 || W >= d11) && continue
            Z = floor(Int, W)
            T =
                d10 - (
                    loggam(Z + 1) +
                    loggam(mingoodbad - Z + 1) +
                    loggam(m - Z + 1) +
                    loggam(maxgoodbad - m + Z + 1)
                )
            (X * (4.0 - X) - 3.0) <= T && break
            (X * (X - T) >= 1) && continue
            (2.0 * log(X) <= T) && break
        end

        if good > bad
            Z = m - Z
        end

        return if m < sample
            good - Z
        else
            Z
        end
    end

    return if sample > 10
        hypergeometric_hrua(good, bad, sample)
    else
        hypergeometric_hyp(good, bad, sample)
    end
end

function check_input(
    X::AbstractArray{S},
    Y::AbstractVector{T},
    W::AbstractVector{U},
    max_features::Int,
    max_depth::Int,
    min_samples_leaf::Int,
    min_samples_split::Int,
    min_purity_increase::Float64,
) where {S,T,U}
    if X isa AbstractMatrix
        n_samples, n_features = size(X)
    else
        n_samples = length(X)
        n_features = 1
    end
    if length(Y) != n_samples
        throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")
    elseif length(W) != n_samples
        throw("dimension mismatch between X and W ($(size(X)) vs $(size(W))")
    elseif max_depth < -1
        throw(
            "unexpected value for max_depth: $(max_depth) (expected:" *
            " max_depth >= 0, or max_depth = -1 for infinite depth)",
        )
    elseif n_features < max_features
        throw(
            "number of features $(n_features) is less than the number " *
            "of max features $(max_features)",
        )
    elseif max_features < 0
        throw("number of features $(max_features) must be >= zero ")
    elseif min_samples_leaf < 1
        throw(
            "min_samples_leaf must be a positive integer " * "(given $(min_samples_leaf))"
        )
    elseif min_samples_split < 2
        throw("min_samples_split must be at least 2 " * "(given $(min_samples_split))")
    end
end



######### tree module ###############

# find an optimal split that satisfy the given constraints
# (max_depth, min_samples_split, min_purity_increase)
function _split!(
    X::AbstractMatrix{S},   # the feature array
    Y::AbstractVector{Int}, # the label array
    W::AbstractVector{U},   # the weight vector
    purity_function::Function,
    node::NodeMeta{S}, # the node to split
    max_features::Int,         # number of features to consider
    max_depth::Int,            # the maximum depth of the resultant tree
    min_samples_leaf::Int,            # the minimum number of samples each leaf needs to have
    min_samples_split::Int,           # the minimum number of samples in needed for a split
    min_purity_increase::Float64,     # minimum purity needed for a split
    indX::AbstractVector{Int}, # an array of sample indices, 1:n_samples
    # we split using samples in indX[node.region]
    # the six arrays below are given for optimization purposes
    nc::AbstractVector{U},    # nc maintains a dictionary of all labels in the samples
    ncl::AbstractVector{U},   # ncl maintains the counts of labels on the left
    ncr::AbstractVector{U},   # ncr maintains the counts of labels on the right
    Xf::AbstractVector{S},
    Yf::AbstractVector{Int},
    Wf::AbstractVector{U},  #build_tree, Wf = nothing 
    rng::Random.AbstractRNG,
) where {S,U}
    region = node.region
    n_samples = length(region)
    n_classes = length(nc)
    nc[:] .= zero(U)
    @simd for i in region
        @inbounds nc[Y[indX[i]]] += W[indX[i]]
    end
    nt = sum(nc)
    node.label = argmax(nc)
    # node.node_impurity = nt * purity_function(nc, nt)
    node.node_impurity = nt - maximum(nc)
    if (
        min_samples_leaf * 2 > n_samples ||
        min_samples_split > n_samples ||
        max_depth <= node.depth ||
        nc[node.label] == nt
    )
        node.is_leaf = true
        return nothing
    end

    r_start = region.start - 1
    features = node.features
    n_features = length(features)
    best_purity = typemin(U)
    best_feature = -1
    eveloss = -Inf * ones(Int, n_features)
    threshold_lo = X[1]
    threshold_hi = X[1]
    be_purity = -Inf * ones(Float64, n_features)


    indf = 1
    # the number of new constants found during this split
    n_const = 0
    # true if every feature is constant
    unsplittable = true
    # the number of non constant features we will see if
    # only sample n_features used features
    # is a hypergeometric random variable
    total_features = size(X, 2)
    # this is the total number of features that we expect to not
    # be one of the known constant features. since we know exactly
    # what the non constant features are, we can sample at 'non_consts_used'
    # non constant features instead of going through every feature randomly.
    non_consts_used = hypergeometric(
        n_features, total_features - n_features, max_features, rng
    )
    k = 0
    @inbounds while (unsplittable || indf <= non_consts_used) && indf <= n_features
        feature = let
            indr = rand(rng, indf:n_features)
            features[indf], features[indr] = features[indr], features[indf]
            features[indf]
            k += 1
        end

        # in the begining, every node is on right of the threshold
        ncl[:] .= zero(U)
        ncr[:] = nc
        @simd for i in 1:n_samples
            Xf[i] = X[indX[i+r_start], feature]
        end

        # sort Yf and indX by Xf
        q_bi_sort!(Xf, indX, 1, n_samples, r_start)
        @simd for i in 1:n_samples
            Yf[i] = Y[indX[i+r_start]]
            Wf[i] = W[indX[i+r_start]]
        end

        hi = 0
        nl, nr = zero(U), nt
        is_constant = true
        last_f = Xf[1]
        delta = 0
        while hi < n_samples
            lo = hi + 1
            curr_f = Xf[lo]


            (lo != 1) && (is_constant = false)
            # honor min_samples_leaf
            # if nl >= min_samples_leaf && nr >= min_samples_leaf
            # @assert nl == lo-1,
            # @assert nr == n_samples - (lo-1) == n_samples - lo + 1

            if lo - 1 >= min_samples_leaf && n_samples - (lo - 1) >= min_samples_leaf
                unsplittable = false
                 # purity = -(nl * purity_function(ncl, nl) + nr * purity_function(ncr, nr)) # original objective function
                    purity = -(nt - maximum(ncl)  - maximum(ncr)) # misclassification error
                if purity > be_purity[feature]
                    be_purity[feature] = purity
                    eveloss[feature] = purity
                end
                if purity > best_purity && !isapprox(purity, best_purity)
                    threshold_lo = last_f
                    threshold_hi = curr_f
                    best_purity = purity
                    best_feature = feature
                end
                delta = max(Int(best_purity - purity), 0) #delta = 0 to delete reduction
            end
            indnext = min(lo + delta, n_samples)
            ind_jump = searchsortedlast(Xf, Xf[indnext], indnext, n_samples, Base.Order.Forward)
            hi = ind_jump
            
            # fill ncl and ncr in the direction
            # that would require the smaller number of iterations
            # i.e., hi - lo < n_samples - hi

            if (hi << 1) < n_samples + lo # ncr: number of each class exists in right set
                @simd for i in lo:hi
                    ncr[Yf[i]] -= Wf[i]
                end
            else
                ncr[:] .= zero(U)
                @simd for i in (hi+1):n_samples
                    ncr[Yf[i]] += Wf[i]
                end
            end

            nr = zero(U)
            @simd for lab in 1:n_classes
                nr += ncr[lab]  # nr: number of samples in right set
                ncl[lab] = nc[lab] - ncr[lab] #ncr: number of each class exists in left set
            end
            nl = nt - nr
            last_f = Xf[hi]#curr_f
        end
        #println("Feature: ", feature, " Iter: ", iter, " k: ", k)
        # keep track of constant features to be used later.
        if is_constant
            n_const += 1
            features[indf], features[n_const] = features[n_const], features[indf]
        end

        indf += 1
    end

    # no splits honor min_samples_leaf
    @inbounds if (
        unsplittable || (best_purity + node.node_impurity < min_purity_increase * nt)
    )
        node.is_leaf = true
        return nothing   ### stop as a leaf node
    else
        @simd for i in 1:n_samples
            Xf[i] = X[indX[i+r_start], best_feature]
        end

        try
            node.threshold = (threshold_lo + threshold_hi) / 2.0
            node.threbound = [threshold_lo, threshold_hi]
        catch
            node.threshold = threshold_hi
            node.threbound = [threshold_hi, threshold_hi]
        end
        # split the samples into two parts: ones that are greater than
        # the threshold and ones that are less than or equal to the threshold
        #                                 ---------------------
        # (so we partition at threshold_lo instead of node.threshold)
        node.split_at = partition!(indX, Xf, threshold_lo, region)
        node.feature = best_feature
        node.eveloss = eveloss
        node.features = features[(n_const+1):n_features]
    end
    return _split!
end
@inline function fork!(node::NodeMeta{S}) where {S}
    ind = node.split_at
    region = node.region
    features = node.features
    # no need to copy because we will copy at the end
    node.l = NodeMeta{S}(features, region[1:ind], node.depth + 1)
    node.r = NodeMeta{S}(features, region[(ind+1):end], node.depth + 1)
end

function _fit(
    X::AbstractMatrix{S},
    Y::AbstractVector{Int},
    W::AbstractVector{U},
    loss::Function,
    n_classes::Int,
    max_features::Int,
    max_depth::Int,
    min_samples_leaf::Int,
    min_samples_split::Int,
    min_purity_increase::Float64,
    rng=Random.GLOBAL_RNG::Random.AbstractRNG,
) where {S,U}
    n_samples, n_features = size(X)
    nc = Array{U}(undef, n_classes)
    ncl = Array{U}(undef, n_classes)
    ncr = Array{U}(undef, n_classes)
    Wf = Array{U}(undef, n_samples)
    Xf = Array{S}(undef, n_samples)
    Yf = Array{Int}(undef, n_samples)
    indX = collect(1:n_samples)
    root = NodeMeta{S}(collect(1:n_features), 1:n_samples, 0)
    stack = NodeMeta{S}[root]
    @inbounds while length(stack) > 0
        node = pop!(stack)
        _split!(
            X,
            Y,
            W,
            loss,
            node,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            indX,
            nc,
            ncl,
            ncr,
            Xf,
            Yf,
            Wf,
            rng,
        )
        if !node.is_leaf
            fork!(node)
            push!(stack, node.r)
            push!(stack, node.l)
        end
    end

    return (root, indX)
end

function fit(;
    X::AbstractMatrix{S},
    Y::AbstractVector{T},
    W::Union{Nothing,AbstractVector{U}},
    loss=normal_loss::Function,
    max_features::Int,
    max_depth::Int,
    min_samples_leaf::Int,
    min_samples_split::Int,
    min_purity_increase::Float64,
    rng=Random.GLOBAL_RNG::Random.AbstractRNG,
) where {S,T,U}
    n_samples, n_features = size(X)
    list, Y_ = assign(Y)
    if isnothing(W)
        W = fill(1, n_samples)
    end

    check_input(
        X,
        Y,
        W,
        max_features,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
    )

    root, indX = _fit(
        X,
        Y_,
        W,
        loss,
        length(list),
        max_features,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        rng,
    )

    return Tree{S,T}(root, list, indX)
end

end # module
