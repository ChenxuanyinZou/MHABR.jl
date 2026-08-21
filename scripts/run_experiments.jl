# Usage: julia --project=. scripts/run_experiments.jl ds_start ds_end run_start run_end time_limit tree_depth epsilon batch flag

using Random, StatsBase, CSV, DataFrames
using MHABR

function load_dataset(dataset::Int, run::Int, dir_path::AbstractString)
    data_prefix = string(dataset, "_", run, "_")
    read_split(split) = Matrix(DataFrame(CSV.File(
        joinpath(dir_path, data_prefix * split);
        header=false,
        missingstring="?",
    )))
    return read_split("train"), read_split("val"), read_split("test")
end

experiment_runs(run_start::Int, run_end::Int) = collect(run_start:run_end)
result_column(run::Int, run_start::Int) = run - run_start + 1

function result_filename(
    dataset_start::Int,
    dataset_end::Int,
    time_limit::Float64,
    run_start::Int,
    run_end::Int,
    tree_depth::Int,
    epsilon::Float64,
    metric::AbstractString,
)
    return string(
        "CART_MH_", dataset_start, "_", dataset_end,
        "_time_limit_", time_limit,
        "_runs_", run_start, "_", run_end,
        "_d_", tree_depth, "_", epsilon,
        "__Nmin_1_", metric, "_updated.csv",
    )
end

function write_parameterized_results(
    dataset_start::Int,
    dataset_end::Int,
    time_limit::Float64,
    run_start::Int,
    run_end::Int,
    tree_depth::Int,
    epsilon::Float64,
    train_accs::Matrix{Float64},
    test_accs::Matrix{Float64},
    train_times::Matrix{Float64},
)
    outputs = (
        ("training_accuracys", train_accs),
        ("test_accuracys", test_accs),
        ("training_times", train_times),
    )
    for (metric, values) in outputs
        CSV.write(
            result_filename(
                dataset_start,
                dataset_end,
                time_limit,
                run_start,
                run_end,
                tree_depth,
                epsilon,
                metric,
            ),
            DataFrame(values, :auto),
        )
    end
    return nothing
end

function main(args=ARGS)
    if length(args) != 9
        println(stderr,
            "Usage: julia --project=. scripts/run_experiments.jl <ds_start> <ds_end> " *
            "<run_start> <run_end> <time_limit> <tree_depth> <epsilon> <batch> <flag>")
        return 1
    end

    dir_path = "./data/"
    dataset_start = parse(Int, args[1])
    dataset_end = parse(Int, args[2])
    run_start = parse(Int, args[3])
    run_end = parse(Int, args[4])
    time_limit = parse(Float64, args[5])
    tree_depth = parse(Int, args[6])
    epsilon = parse(Float64, args[7])
    batch = parse(Float64, args[8])
    alpha_active = parse(Float64, args[9])

    if alpha_active == 0
        alpha_set = [0.0]
    else
        alpha_set = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        @warn "Legacy reproduction mode selects alpha using the test set. " *
              "Do not use this mode for new unbiased evaluations."
    end

    len = run_end - run_start + 1
    result_train = zeros(100, len + 2)
    result_test = zeros(100, len + 2)
    train_accs = zeros(100, len + 2)
    test_accs = zeros(100, len + 2)
    split_num = zeros(100, len + 2)
    train_times = zeros(100, len + 2)

    for dataset in dataset_start:dataset_end
        println("######## Dataset: ", dataset, " ########")
        for run in experiment_runs(run_start, run_end)
            column = result_column(run, run_start)
            start = time()
            println("## Dataset: ", dataset, " Run: ", run, " ##")

            train_data, validation_data, test_data = load_dataset(dataset, run, dir_path)
            p = size(train_data, 2) - 1
            X = vcat(train_data[:, 1:p], validation_data[:, 1:p])
            Y = Int.(vcat(train_data[:, p+1], validation_data[:, p+1]))
            X_test = test_data[:, 1:p]
            Y_test = Int.(test_data[:, p+1])

            classes = sort(unique(Y))
            class_labels = length(classes)
            encoded_Y = zeros(Int, length(Y))
            encoded_Y_test = zeros(Int, length(Y_test))
            for i in 1:class_labels
                encoded_Y[findall(==(classes[i]), Y)] .= i
                encoded_Y_test[findall(==(classes[i]), Y_test)] .= i
            end
            Y = encoded_Y
            Y_test = encoded_Y_test
            println(
                "dataset: ", dataset,
                " run: ", run,
                " n_train: ", size(X, 1),
                " n: ", size(vcat(X, X_test), 1),
                " p: ", p,
                " class_labels: ", class_labels,
            )

            opt_train = 0
            opt_test = Inf
            number_splits = zeros(length(alpha_set))
            best_index = 0

            for (index, alpha_i) in enumerate(alpha_set)
                fit_rng = MersenneTwister(dataset * 1_000_000 + run * 100 + index)
                A, B, C, _ = mh(
                    X, Y, tree_depth, epsilon, alpha_i, batch;
                    rng=fit_rng, time_limit=time_limit,
                )
                println(" A: ", A, " B: ", B, " C: ", C)
                C, train_cost = MHABR.Utils.route(X, Y, A, B)
                test_cost = MHABR.Utils.eva(X_test, Y_test, A, B, C)
                splits = count(!iszero, A)
                println(" Alpha: ", alpha_i, " number of splits: ", splits)
                number_splits[index] = splits
                if test_cost < opt_test
                    opt_train = train_cost
                    opt_test = test_cost
                    best_index = index
                end
            end

            num_splits = number_splits[best_index]
            train_acc = 1 - opt_train / size(X, 1)
            test_acc = 1 - opt_test / size(X_test, 1)
            train_acc = train_acc * 100
            test_acc = test_acc * 100
            split_num[dataset, column] = num_splits
            train_accs[dataset, column] = train_acc
            test_accs[dataset, column] = test_acc
            end_time = time()
            train_times[dataset, column] = end_time - start
            println(
                "Train Acc: ", train_acc,
                " Train_cost ", opt_train,
                " Test Acc: ", test_acc,
                " Time: ", end_time - start,
                " Splits: ", num_splits,
            )
            result_train[dataset, column] = train_acc
            result_test[dataset, column] = test_acc
            write_parameterized_results(
                dataset_start,
                dataset_end,
                time_limit,
                run_start,
                run_end,
                tree_depth,
                epsilon,
                train_accs,
                test_accs,
                train_times,
            )
        end
    end

    train_accs[:, len+1] = mean(train_accs[:, 1:len], dims=2)
    train_accs[:, len+2] = std(train_accs[:, 1:len], dims=2)
    test_accs[:, len+1] = mean(test_accs[:, 1:len], dims=2)
    test_accs[:, len+2] = std(test_accs[:, 1:len], dims=2)
    train_times[:, len+1] = mean(train_times[:, 1:len], dims=2)
    train_times[:, len+2] = std(train_times[:, 1:len], dims=2)
    split_num[:, len+1] = mean(split_num[:, 1:len], dims=2)
    split_num[:, len+2] = std(split_num[:, 1:len], dims=2)

    println("Mean Train Acc: ", train_accs[:, len+1])
    println("Mean Test Acc: ", test_accs[:, len+1])
    println("Mean Train Time: ", train_times[:, len+1])
    result_train[:, len+1] = mean(result_train[:, 1:len], dims=2)
    result_test[:, len+1] = mean(result_test[:, 1:len], dims=2)
    CSV.write("Train_Result", DataFrame(result_train, :auto))
    CSV.write("Train_Test", DataFrame(result_test, :auto))
    write_parameterized_results(
        dataset_start,
        dataset_end,
        time_limit,
        run_start,
        run_end,
        tree_depth,
        epsilon,
        train_accs,
        test_accs,
        train_times,
    )

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
