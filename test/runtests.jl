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
    @test isdefined(MHABR, :load_iris)
end

@testset "Bundled Iris dataset" begin
    X, y = MHABR.load_iris()
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    train = vcat(1:40, 51:90, 101:140)
    test = vcat(41:50, 91:100, 141:150)

    @test X isa Matrix{Float64}
    @test y isa Vector{String}
    @test size(X) == (150, 4)
    @test length(y) == 150
    @test all(isfinite, X)
    @test sort(unique(y)) == classes
    @test all(label -> count(==(label), y) == 50, classes)
    @test length(train) == 120
    @test length(test) == 30
    @test all(label -> count(==(label), y[train]) == 40, classes)
    @test all(label -> count(==(label), y[test]) == 10, classes)

    mktempdir() do dir
        missing_path = joinpath(dir, "missing.csv")
        missing_error = try
            MHABR._load_iris(missing_path)
            nothing
        catch error
            error
        end
        @test missing_error isa ErrorException
        @test occursin(missing_path, sprint(showerror, missing_error))

        bad_size_path = joinpath(dir, "bad-size.csv")
        write(bad_size_path, "1,2,3,4,Iris-setosa\n")
        @test_throws ErrorException MHABR._load_iris(bad_size_path)

        iris_path = joinpath(@__DIR__, "..", "examples", "data", "iris.csv")
        bad_classes_path = joinpath(dir, "bad-classes.csv")
        write(
            bad_classes_path,
            replace(read(iris_path, String), "Iris-setosa" => "Iris-unknown"),
        )
        @test_throws ErrorException MHABR._load_iris(bad_classes_path)
    end
end

include(joinpath(@__DIR__, "..", "scripts", "run_experiments.jl"))

@testset "Experiment runner" begin
    @test experiment_runs(1, 4) == [1, 2, 3, 4]
    @test result_column(5, 5) == 1
    @test result_column(6, 5) == 2
    @test main(String[]) == 1

    @test result_filename(
        1, 56, 10.0, 1, 10, 4, 0.0, "training_accuracys"
    ) ==
        "CART_MH_1_56_time_limit_10.0_runs_1_10_d_4_0.0__Nmin_1_training_accuracys_updated.csv"
    @test result_filename(
        1, 56, 10.0, 1, 10, 4, 0.0, "test_accuracys"
    ) ==
        "CART_MH_1_56_time_limit_10.0_runs_1_10_d_4_0.0__Nmin_1_test_accuracys_updated.csv"
    @test result_filename(
        1, 56, 10.0, 1, 10, 4, 0.0, "training_times"
    ) ==
        "CART_MH_1_56_time_limit_10.0_runs_1_10_d_4_0.0__Nmin_1_training_times_updated.csv"

    mktempdir() do dir
        data_dir = joinpath(dir, "data")
        mkdir(data_dir)
        rows = "0.0,0.0,1\n1.0,1.0,2\n"
        for split in ("train", "val", "test")
            write(joinpath(data_dir, "1_1_$(split)"), rows)
        end

        cd(dir) do
            @test main(["1", "1", "1", "1", "0", "1", "0", "1", "0"]) == 0

            training_accuracy =
                "CART_MH_1_1_time_limit_0.0_runs_1_1_d_1_0.0__Nmin_1_training_accuracys_updated.csv"
            test_accuracy =
                "CART_MH_1_1_time_limit_0.0_runs_1_1_d_1_0.0__Nmin_1_test_accuracys_updated.csv"
            training_time =
                "CART_MH_1_1_time_limit_0.0_runs_1_1_d_1_0.0__Nmin_1_training_times_updated.csv"
            expected_files = sort([
                "Train_Result",
                "Train_Test",
                training_accuracy,
                test_accuracy,
                training_time,
            ])

            files = filter(isfile, readdir(dir; join=true))
            @test sort(basename.(files)) == expected_files
            for path in (training_accuracy, test_accuracy, training_time)
                @test size(DataFrame(CSV.File(path))) == (100, 3)
            end
        end
    end
end
