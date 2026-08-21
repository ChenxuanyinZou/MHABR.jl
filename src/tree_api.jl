abstract type AbstractMHABRNode{L} end

struct SplitNode{L} <: AbstractMHABRNode{L}
    feature::Int
    threshold::Float64
    left::AbstractMHABRNode{L}
    right::AbstractMHABRNode{L}
end

struct LeafNode{L} <: AbstractMHABRNode{L}
    prediction::Union{L,Missing}
    sample_count::Int
    class_counts::Dict{L,Int}
end

struct MHABRTree{L}
    root::AbstractMHABRNode{L}
    classes::Vector{L}
    n_features::Int
    max_depth::Int
    training_loss::Int
    legacy_parameters::NamedTuple{
        (:A, :B, :C),
        Tuple{Vector{Int},Vector{Float64},Vector{Int}},
    }
end

function _encode_labels(y::AbstractVector{L}) where {L}
    isempty(y) && throw(ArgumentError("y must not be empty"))
    any(ismissing, y) && throw(ArgumentError("y must not contain missing labels"))

    classes = collect(unique(y))
    if L <: Integer && sort(Int.(classes)) == collect(1:length(classes))
        sort!(classes)
    end
    label_to_code = Dict(label => code for (code, label) in enumerate(classes))
    encoded = [label_to_code[label] for label in y]
    return classes, encoded
end

function _validated_features(X::AbstractMatrix)
    size(X, 2) > 0 || throw(ArgumentError("X must contain at least one feature"))
    all(value -> value isa Real && !ismissing(value) && isfinite(value), X) ||
        throw(ArgumentError("X must contain only finite, non-missing real values"))
    return Float64.(X)
end

function _class_counts(y::AbstractVector{L}, rows::AbstractVector{Int}) where {L}
    counts = Dict{L,Int}()
    for row in rows
        label = y[row]
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

function _leaf_prediction(
    C::Vector{Int},
    classes::Vector{L},
    node_index::Int,
    branch_size::Int,
) where {L}
    terminal = node_index
    while terminal <= branch_size
        terminal = 2 * terminal + 1
    end
    code = C[terminal - branch_size]
    return code == 0 ? missing : classes[code]
end

function _build_recursive_node(
    A::Vector{Int},
    B::Vector{Float64},
    C::Vector{Int},
    X::Matrix{Float64},
    y::AbstractVector{L},
    classes::Vector{L},
    rows::Vector{Int},
    node_index::Int,
) where {L}
    branch_size = length(A)
    if node_index > branch_size || A[node_index] == 0
        prediction = _leaf_prediction(C, classes, node_index, branch_size)
        return LeafNode{L}(prediction, length(rows), _class_counts(y, rows))
    end

    feature = A[node_index]
    threshold = B[node_index]
    left_rows = [row for row in rows if X[row, feature] < threshold]
    right_rows = [row for row in rows if X[row, feature] >= threshold]
    left = _build_recursive_node(A, B, C, X, y, classes, left_rows, 2 * node_index)
    right = _build_recursive_node(A, B, C, X, y, classes, right_rows, 2 * node_index + 1)
    return SplitNode{L}(feature, threshold, left, right)
end

function mhabr(
    X::AbstractMatrix,
    y::AbstractVector{L};
    max_depth::Int,
    epsilon::Real=0.0,
    alpha::Real=0.0,
    batch::Real=1.0,
    rng::Random.AbstractRNG=Random.default_rng(),
    time_limit::Real=Inf,
) where {L}
    size(X, 1) == length(y) || throw(DimensionMismatch("X rows must match y length"))
    X_float = _validated_features(X)
    classes, encoded = _encode_labels(y)
    A, B, C, loss = mh(
        X_float,
        encoded,
        max_depth,
        Float64(epsilon),
        Float64(alpha),
        Float64(batch);
        rng,
        time_limit,
    )
    rows = collect(axes(X_float, 1))
    root = _build_recursive_node(A, B, C, X_float, y, classes, rows, 1)
    legacy = (A=copy(A), B=copy(B), C=copy(C))
    return MHABRTree{L}(root, classes, size(X_float, 2), max_depth, loss, legacy)
end

fit(X::AbstractMatrix, y::AbstractVector; kwargs...) = mhabr(X, y; kwargs...)

_predict(node::LeafNode, row) = node.prediction
function _predict(node::SplitNode, row)
    child = row[node.feature] < node.threshold ? node.left : node.right
    return _predict(child, row)
end

function _validate_prediction_features(tree::MHABRTree, X::AbstractMatrix)
    size(X, 2) == tree.n_features || throw(DimensionMismatch(
        "X has $(size(X, 2)) features; expected $(tree.n_features)",
    ))
    return _validated_features(X)
end

function predict(tree::MHABRTree{L}, X::AbstractMatrix) where {L}
    X_float = _validate_prediction_features(tree, X)
    predictions = Vector{Union{L,Missing}}(undef, size(X_float, 1))
    for row in axes(X_float, 1)
        predictions[row] = _predict(tree.root, @view X_float[row, :])
    end
    return predictions
end

_leaf_for(node::LeafNode, row) = node
function _leaf_for(node::SplitNode, row)
    child = row[node.feature] < node.threshold ? node.left : node.right
    return _leaf_for(child, row)
end

function predict_proba(tree::MHABRTree, X::AbstractMatrix)
    X_float = _validate_prediction_features(tree, X)
    probabilities = Matrix{Float64}(undef, size(X_float, 1), length(tree.classes))
    for row in axes(X_float, 1)
        leaf = _leaf_for(tree.root, @view X_float[row, :])
        if leaf.sample_count == 0
            probabilities[row, :] .= NaN
        else
            for (column, label) in enumerate(tree.classes)
                probabilities[row, column] = get(leaf.class_counts, label, 0) / leaf.sample_count
            end
        end
    end
    return probabilities
end

function score(tree::MHABRTree, X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X rows must match y length"))
    isempty(y) && throw(ArgumentError("y must not be empty"))
    predictions = predict(tree, X)
    correct = count(
        index -> !ismissing(predictions[index]) && predictions[index] == y[index],
        eachindex(y),
    )
    return correct / length(y)
end

function parameters(tree::MHABRTree)
    stored = tree.legacy_parameters
    return copy(stored.A), copy(stored.B), copy(stored.C)
end

function _print_node(io::IO, node::LeafNode, indent::String)
    println(io, indent, "Leaf(prediction=", node.prediction,
        ", samples=", node.sample_count, ", counts=", node.class_counts, ")")
end

function _print_node(io::IO, node::SplitNode, indent::String)
    println(io, indent, "feature[", node.feature, "] < ", node.threshold)
    _print_node(io, node.left, indent * "  L: ")
    _print_node(io, node.right, indent * "  R: ")
end

function print_tree(io::IO, tree::MHABRTree)
    _print_node(io, tree.root, "")
    return nothing
end
print_tree(tree::MHABRTree) = print_tree(stdout, tree)

_split_count(::LeafNode) = 0
_split_count(node::SplitNode) = 1 + _split_count(node.left) + _split_count(node.right)
_leaf_count(::LeafNode) = 1
_leaf_count(node::SplitNode) = _leaf_count(node.left) + _leaf_count(node.right)

function Base.show(io::IO, tree::MHABRTree)
    print(io, "MHABRTree(depth=", tree.max_depth,
        ", splits=", _split_count(tree.root),
        ", leaves=", _leaf_count(tree.root),
        ", classes=", length(tree.classes),
        ", training_loss=", tree.training_loss, ")")
end
