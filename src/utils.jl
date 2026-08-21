module Utils

using ..ClassificationTree
using StatsBase

export get_ab, get_c, get_data, route, eva

function get_ab(
    dc::Int, 
    D::Int, 
    a::Vector{Int}, 
    b::Vector{Float64}, 
    start_ind::Int, 
    root_tree
)
    stack = Tuple{Any, Int}[]
    push!(stack, (root_tree, start_ind))
    while !isempty(stack)
        current_node, ind = pop!(stack)
        if ind > length(a)
            continue
        end
        if hasproperty(current_node, :featid)
            a[ind] = current_node.featid
            b[ind] = convert(Float64, current_node.featval)
            push!(stack, (current_node.right, 2 * ind + 1))
            push!(stack, (current_node.left, 2 * ind))
        elseif hasproperty(current_node, :node)
            push!(stack, (current_node.node, ind))
        else
            a[ind] = 0
            b[ind] = 0.0
            push!(stack, (current_node, 2 * ind))
            push!(stack, (current_node, 2 * ind + 1))
        end
    end
    
    return a, b
end

function get_c(
    X::AbstractMatrix{Float64}, 
    Y::AbstractVector{Int}, 
    A::Vector, 
    B::Vector
)  #get the class labels of the leaf nodes
    n, p = size(X)
    branch_size = size(A, 1)
    leaf_size = size(A, 1) + 1
    Z = zeros(Int, n)
    for i in 1:n
        node = 1
        while node <= branch_size
            if A[node] == 0
                node = 2 * node + 1
                continue
            end
            if X[i, A[node]] < B[node]
                node = 2 * node
            else
                node = 2 * node + 1
            end
        end
        Z[i] = node - branch_size
    end
    C = zeros(Int, leaf_size)
    for i in 1:leaf_size
        tmp_Z = Z .== i
        if sum(tmp_Z) == 0
            continue
        end
        tmp_Y = Y[tmp_Z]
        C[i] = mode(tmp_Y)
    end

    return C
end

function get_data(
    X::AbstractMatrix{Float64}, 
    Y::AbstractVector{Int}, 
    i::Int, 
    A::Vector, 
    B::Vector
)   #get the data partitioned to each node
    n, p = size(X)
    ancesters = [floor(Int32, i / 2^j) for j in floor(Int32, log2(i)):-1:0] #the routing path
    size_ancs = size(ancesters, 1)
    selected = trues(n)
    for index in 1:n
        for i in 1:(size_ancs-1) # 1,2
            if ancesters[i+1] == 2 * ancesters[i] # left_node, ax<b
                if A[ancesters[i]] == 0
                    selected[index] = false
                    continue
                end
                if selected[index] #  X[index, :][a[:, ancesters[i]]][1] >= b[ancesters[i]]
                    if X[index, A[ancesters[i]]] >= B[ancesters[i]]
                        selected[index] = false
                    end     # end if
                end         # end if 
            else            # right_node, ax>=b
                if A[ancesters[i]] == 0
                    continue
                end
                if selected[index] #  X[index, :][a[:, ancesters[i]]][1] < b[ancesters[i]]
                    if X[index, A[ancesters[i]]] < B[ancesters[i]]
                        selected[index] = false
                    end      # end if
                end          # end if 
            end              # end if         
        end                  # end for
    end                      # end if
    return copy(X[selected, :]), copy(Y[selected])
end



function route(
    X::AbstractArray{S}, 
    Y::AbstractArray{Int}, 
    A::Vector, 
    B::Vector
) where{S}
    n, p = size(X)
    branch_size = size(A, 1)
    leaf_size = branch_size + 1
    Z = zeros(Int, n)
    num_error = 0
    C = zeros(Int, leaf_size)

    # Determine the leaf node for each sample and evaluate the error
    for i in 1:n
        node = 1
        while node <= branch_size
            if A[node] == 0
                node = 2 * node + 1
                continue
            end
            node = ifelse(X[i, A[node]] < B[node], 2 * node, 2 * node + 1)
        end
        leaf_node = node - branch_size
        Z[i] = leaf_node
    end

    # Compute the class for each leaf node
    n_samples = copy(C)
    for node_leaf in 1:leaf_size
        tmp_Y = Y[Z.==node_leaf]
        n_samples[node_leaf] = length(tmp_Y)
        if !isempty(tmp_Y)
            C[node_leaf] = mode(tmp_Y)
            num_error += sum(tmp_Y .!= C[node_leaf])
        end
    end

    return C, num_error#, n_samples
end


function eva(
    X::AbstractMatrix{Float64}, 
    Y::AbstractVector{Int}, 
    A::Vector, 
    B::Vector, 
    C::Vector
) #evaluate the misclassigication loss of the tree
    n, p = size(X)
    branch_size = size(A, 1)
    num_error = 0
    for i in 1:n
        node = 1
        while node <= branch_size
            feature = A[node]
            if feature == 0
                node = 2 * node + 1
                continue
            end
            if X[i, feature] < B[node]
                node = 2 * node
            else
                node = 2 * node + 1
            end
        end

        if Y[i] != C[node - branch_size]
            num_error += 1
        end
    end
    return num_error
end

end # module







