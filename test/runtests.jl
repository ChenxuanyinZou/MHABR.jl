using Test
using MHABR
using Random

@testset "Legacy array algorithm" begin
    X = reshape([0.0, 1.0], :, 1)
    y = [1, 2]

    A, B, C, loss = mh(X, y, 1)

    @test A == [1]
    @test B == [0.5]
    @test C == [1, 2]
    @test loss == 0

    A0, B0, C0, loss0 = mh(X, y, 1; time_limit=0.0)
    @test A0 == [0]
    @test B0 == [0.0]
    @test C0 == [0, 1]
    @test loss0 == 1

    @test_throws ArgumentError mh(X, y, 1, 0.0, 0.0, 0.0)

    X_batch = reshape(collect(0.0:7.0), :, 1)
    y_batch = [1, 2, 1, 2, 1, 2, 1, 2]
    result1 = mh(X_batch, y_batch, 2, 0.0, 0.0, 0.5; rng=MersenneTwister(42))
    result2 = mh(X_batch, y_batch, 2, 0.0, 0.0, 0.5; rng=MersenneTwister(42))
    @test result1 == result2
end

@testset "Recursive tree API" begin
    X = reshape([0.0, 1.0], :, 1)
    labels = ["left", "right"]
    tree = fit(X, labels; max_depth=1, rng=MersenneTwister(7))

    @test tree isa MHABRTree{String}
    @test tree.root isa SplitNode{String}
    @test predict(tree, X) == labels
    @test score(tree, X, labels) == 1.0
    @test predict_proba(tree, X) == [1.0 0.0; 0.0 1.0]
    @test parameters(tree) == ([1], [0.5], [1, 2])

    A, _, _ = parameters(tree)
    A[1] = 99
    @test parameters(tree)[1] == [1]

    left = tree.root.left
    right = tree.root.right
    @test left isa LeafNode{String}
    @test right isa LeafNode{String}
    @test left.sample_count == 1
    @test left.class_counts == Dict("left" => 1)
    @test right.sample_count == 1
    @test right.class_counts == Dict("right" => 1)

    tree_text = sprint(print_tree, tree)
    @test occursin("feature[1] < 0.5", tree_text)
    @test occursin("Leaf(prediction=left, samples=1", tree_text)
    @test sprint(show, tree) ==
        "MHABRTree(depth=1, splits=1, leaves=2, classes=2, training_loss=0)"

    symbol_tree = mhabr(X, [:a, :b]; max_depth=1)
    @test predict(symbol_tree, X) == [:a, :b]

    folded = fit(X, labels; max_depth=1, time_limit=0.0)
    @test folded.root isa LeafNode{String}
    @test folded.root.sample_count == 2

    @test_throws DimensionMismatch fit(zeros(3, 1), labels; max_depth=1)
    @test_throws ArgumentError fit([0.0; NaN;;], labels; max_depth=1)
    @test_throws ArgumentError fit(X, labels; max_depth=0)
end

@testset "Package API smoke test" begin
    @test isdefined(MHABR, :MHABRTree)
    @test isdefined(MHABR, :fit)
    @test isdefined(MHABR, :predict)
end

include(joinpath(@__DIR__, "..", "scripts", "run_experiments.jl"))

@testset "Experiment runner" begin
    @test experiment_runs(1, 4) == [1, 2, 3, 4]
    @test result_column(5, 5) == 1
    @test result_column(6, 5) == 2
    @test main(String[]) == 1
end
