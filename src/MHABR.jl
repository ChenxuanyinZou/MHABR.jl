module MHABR

include("ClassificationTree.jl") # R-CART implementation
using .ClassificationTree
using StatsBase: mode
import StatsBase: fit, predict, score
using Random
include("utils.jl") # utility functions, including data loading and processing
using .Utils


export AbstractMHABRNode, SplitNode, LeafNode, MHABRTree
export fit, mhabr, predict, predict_proba, score, parameters, print_tree
export mh


function mh(
    X::AbstractMatrix{Float64}, 
    Y::AbstractVector{Int}, 
    tree_depth::Int, 
    epsilon=0.0::Float64, 
    alpha=0.0::Float64, 
    batch=1.0::Float64;
    rng::Random.AbstractRNG=Random.default_rng(),
    time_limit::Real=Inf
)
    tree_depth >= 1 || throw(ArgumentError("tree_depth must be at least 1"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    alpha >= 0 || throw(ArgumentError("alpha must be nonnegative"))
    0 < batch <= 1 || throw(ArgumentError("batch must be in (0, 1]"))
    time_limit >= 0 || throw(ArgumentError("time_limit must be nonnegative"))
    size(X, 1) == length(Y) || throw(DimensionMismatch("X rows must match Y length"))

    branch_size, leaf_size = 2^tree_depth - 1, 2^tree_depth
    if length(Y) == 0 || size(X, 1) == 0
        return zeros(Int, branch_size), zeros(Float64, branch_size), zeros(Int, leaf_size), 0
    end
    best_A, best_B = zeros(Int, branch_size), zeros(Float64, branch_size)
    best_C, min_cost = route(X, Y, best_A, best_B)
    deadline = isfinite(time_limit) ? time() + Float64(time_limit) : Inf
    pending = collect(1:max(2^(tree_depth - 1) - 1, 1))
    while !isempty(pending)
        time() >= deadline && break
        node = popfirst!(pending)
        A, B = copy(best_A), copy(best_B)
        subtree_depth = floor(Int32, log2(node)) + 1
        sub_tree_size = 2^(tree_depth - subtree_depth + 1) - 1
        sub_a, sub_b = zeros(Int, sub_tree_size), zeros(Float64, sub_tree_size)
        sub_c = zeros(Int, 2^(tree_depth - subtree_depth + 1))
        X_i, Y_i = get_data(X, Y, node, A, B)
        @inbounds if size(X_i, 1) > 1 && length(unique(Y_i)) > 1 #substitute the node parameters corresponding to the subtree
            subnodes = Int[]
            sub_a, sub_b, sub_c, sub_sign = branch_reduce(
                X_i, Y_i, node, tree_depth, epsilon, alpha, batch; rng, deadline
            )
            for d = 0:(tree_depth-subtree_depth)
                append!(subnodes, 2^d*node:2^d*node+2^d-1)
                idx_range = (2^d*node):(2^d*node+2^d-1)
                A[idx_range] = copy(sub_a[2^d:2^(d+1)-1])
                B[idx_range] = copy(sub_b[2^d:2^(d+1)-1])
            end
            if sub_sign  #if the subtree is optimal, remove the nodes from the mhind
                filter!(x -> x ∉ subnodes, pending)
            end
        end
        C, cost = route(X, Y, A, B)
        if cost === 0 ## if the cost is 0, return the global optimal result
            return A, B, C, 0
        end
        if cost < min_cost
            best_A, best_B, best_C, min_cost = copy(A), copy(B), copy(C), cost
        end
    end
    return best_A, best_B, best_C, min_cost
end

function branch_reduce(
    X_i::AbstractMatrix{Float64}, 
    Y_i::AbstractArray{Int}, 
    node_i::Int, 
    depth::Int, 
    epsilon=0.0::Float64, 
    alpha=0.0::Float64, 
    batch=1.0::Float64;
    rng::Random.AbstractRNG=Random.default_rng(),
    deadline::Float64=Inf
)  # function of the Branch and Reduce method
    signopt = false
    n, p = size(X_i)
    depth_i = floor(Int, log2(node_i)) + 1
    D = depth - depth_i
    _, min_cost, best_A, best_B = evaluate(Y_i, X_i, D + 1, alpha; rng)
    best_C = zeros(Int, 2^(D + 1))
    splits, group = gen_splits(X_i, Y_i)    # generate splits and group
    for i in 1:p
        time() >= deadline && break
        splits_i, group_i = splits[i], group[i]
        n_splits = length(splits_i)
        N_epsilon = epsilon * length(Y_i)
        min_cost, best_A, best_B, best_C = bestsplit(
            min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i,
            collect(1:n_splits), D, i, N_epsilon, alpha, batch; rng, deadline
        )
    end
    if min_cost === 0           # if the cost is 0, the current subtree is the optimal
        signopt = true
    end
    return best_A, best_B, best_C, signopt
end


@inline function bestsplit(
    min_cost::Int,
    best_A::Vector{Int},
    best_B::Vector{Float64},
    best_C::Vector{Int},
    X_i::AbstractMatrix{Float64},
    Y_i::AbstractVector{Int},
    splits_i::Vector{Float64},
    group_i::Vector{Int},
    indset::AbstractVector{Int},
    D::Int,
    ai::Int,
    N_epsilon::Float64,
    alpha::Float64,
    batch::Float64;
    rng::Random.AbstractRNG=Random.default_rng(),
    deadline::Float64=Inf
)
    cur_cost = min_cost                                                                 # min_cost is the upper bound as well as the historical minimum cost
    if time() < deadline && !isempty(indset) && length(indset) > N_epsilon
        k = indset[ceil(Int, length(indset) / 2)]                                       #bisection
        split = splits_i[k]
        X_left = @view X_i[X_i[:, ai].<split, :]                                        # split the data
        Y_left = @view Y_i[X_i[:, ai].<split]
        X_right = @view X_i[X_i[:, ai].>=split, :]
        Y_right = @view Y_i[X_i[:, ai].>=split]
        indset = indset[indset.!=k]
        if (batch < 1.0) & (batch > 0.0)                                                # Randomly pick a batch of the data to evaluate the subtree costs
            a1 = shuffle(rng, 1:length(Y_left))
            a2 = shuffle(rng, 1:length(Y_right))
            ind1, ind2 = a1[1:ceil(Int, length(a1) * batch)], a2[1:ceil(Int, length(a2) * batch)]
            X_left, Y_left = X_left[ind1, :], Y_left[ind1]
            X_right, Y_right = X_right[ind2, :], Y_right[ind2]
        elseif batch > 1.0 || batch <= 0.0
            error("batch is not feasible")
        end
        tree_left, cost_left, left_a, left_b = evaluate(Y_left, X_left, D, alpha; rng)       # evaluate the left subtree
        tree_right, cost_right, right_a, right_b = evaluate(Y_right, X_right, D, alpha; rng) # evaluate the right subtree
        cur_cost = cost_left + cost_right                                               # calculate the cost of the current subtree
        if cur_cost <= min_cost
            best_A[1], best_B[1] = ai, split
            update_best_split!(best_A, best_B, left_a, left_b, right_a, right_b, D)     # update the tree parameters
            min_cost = cur_cost                                                         # update the minimum cost
        end
        delta = max((cur_cost - min_cost), 1)                                            # calculate delta
        left_set = intersect(indset, findall(x -> (x <= group_i[k] - delta), group_i))   # branch the reduced left set
        right_set = intersect(indset, findall(x -> (x >= group_i[k] + delta), group_i))  # branch the reduced right set
        min_cost, best_A, best_B, best_C = bestsplit(
            min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i,
            left_set, D, ai, N_epsilon, alpha, batch; rng, deadline
        )
        min_cost, best_A, best_B, best_C = bestsplit(
            min_cost, best_A, best_B, best_C, X_i, Y_i, splits_i, group_i,
            right_set, D, ai, N_epsilon, alpha, batch; rng, deadline
        )
    end
    return min_cost, best_A, best_B, best_C
end


function evaluate(
    Y::AbstractVector{Int},
    X::AbstractMatrix{Float64},
    D::Int,
    alpha::Float64;
    rng::Random.AbstractRNG=Random.default_rng()
)
    a, b = zeros(Int, 2^D - 1), zeros(Float64, 2^D - 1)
    c = zeros(Int, 2^D)
    if length(Y) >= 1
        if D == 0
            cost = sum(Y .!= mode(Y))
            return nothing, cost, [], []
        else
            tree = ClassificationTree.build_tree(Y, X, 0, D, 1, 2, alpha; rng)   # build the tree with the data
            a, b = get_ab(1, D, a, b, 1, tree.node)
            cost = sum(ClassificationTree.apply_tree(tree, X) .!= Y)
            return tree, cost, a, b
        end
    else
        return nothing, 0, a, b
    end
end

function update_best_split!(
    A::Vector, 
    B::Vector, 
    left_a::Vector, 
    left_b::Vector, 
    right_a::Vector, 
    right_b::Vector, 
    depth_diff::Int
) # update the tree parameters   
    for d in 1:depth_diff
        A[2^d:3*2^(d-1)-1] = left_a[2^(d-1):2^d-1]
        A[3*2^(d-1):2^(d+1)-1] = right_a[2^(d-1):2^d-1]
        B[2^d:3*2^(d-1)-1] = left_b[2^(d-1):2^d-1]
        B[3*2^(d-1):2^(d+1)-1] = right_b[2^(d-1):2^d-1]
    end
end


function gen_splits(
    X::AbstractMatrix{Float64}, 
    Y::AbstractVector{Int}
) #generate the splits of the features
    n, p = size(X)
    splits = Vector{Vector{Float64}}(undef, p)
    group = Vector{Vector{Int}}(undef, p)
    for i in 1:p
        unique_vals = sort(unique(X[:, i]))
        valind = sortperm(X[:, i])
        vals = X[valind, i]
        Yvals = Y[valind]
        n_unique = length(unique_vals)
        groupnumb = zeros(Int, n_unique)
        if n_unique <= 1
            splits[i] = [unique_vals[1]]
            group[i] = [n]
        else
            splits_i = (unique_vals[1:end-1] + unique_vals[2:end]) / 2
            splits_i = vcat(unique_vals[1] - 1.0, splits_i, unique_vals[end] + 1.0)

            for t in 1:n_unique
                groupnumb[t] = searchsortedfirst(vals, unique_vals[t])
            end
            splits[i] = splits_i
            group[i] = groupnumb
        end
    end
    return splits, group
end

include("tree_api.jl")

end # module
