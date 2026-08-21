# Description: This file is used to achieve an Approximate Branch-and-Reduce Method
# How to run: julia --project=. scripts/run_experiments.jl dataset_start dataset_end run_start run_end time_limit tree_depth epsilon batch flag
# dataset_start: the starting index of the dataset (inclusive)
# dataset_end: the ending index of the dataset (inclusive)
# run_start: the starting index of the run (inclusive)
# run_end: the ending index of the run (inclusive)
# time_limit: the time limit for the MH process in seconds
# tree_depth: the maximum depth of the decision tree
# epsilon: the termination condition for the MH process
# batch: the mini-batch size for the MH process
# Example: julia --project=. scripts/run_experiments.jl 1 10 1 10 10 4 0 1 0


using Random, StatsBase, CSV, DataFrames
using MHABR


function loadDataset(dataset, run, dir_path)
    data_path = dir_path * string(dataset) * "_" * string(run) * "_"
    train = DataFrame(CSV.File(data_path * "train", header=false, missingstring="?"))
    train = Matrix(train)
    val = DataFrame(CSV.File(data_path * "val", header=false, missingstring="?"))
    val = Matrix(val)
    test = DataFrame(CSV.File(data_path * "test", header=false, missingstring="?"))
    test = Matrix(test)
    return train, val, test
end

experiment_runs(run_start::Int, run_end::Int) = collect(run_start:run_end)
result_column(run::Int, run_start::Int) = run - run_start + 1

function main(args=ARGS)
    if length(args) != 9
        println(stderr,
            "Usage: julia --project=. scripts/run_experiments.jl <ds_start> <ds_end> " *
            "<run_start> <run_end> <time_limit> <tree_depth> <epsilon> <batch> <flag>")
        return 1
    end

dir_path = "./data/"

#Random.seed!(1) # configure the random seed for reproducibility. You can change the seed value to get different results.
# read args from command line
dataset_start = parse(Int, args[1])
dataset_end = parse(Int, args[2])
run_start = parse(Int, args[3])
run_end = parse(Int, args[4])
time_limit = parse(Float64, args[5])
tree_depth = parse(Int, args[6])
epsilon = parse(Float64, args[7]) #terminate condition-epsilon
batch = parse(Float64, args[8])   #mini-batch size
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
greedy_accs = zeros(100, len + 2)
train_times = zeros(100, len + 2)


for dataset in dataset_start:dataset_end
    println("######## Dataset: ", dataset, " ########")
    for run in experiment_runs(run_start, run_end)
        column = result_column(run, run_start)
        # try
        start = time()
        println("## Dataset: ", dataset, " Run: ", run, " ##")
        train, val, test = loadDataset(dataset, run, dir_path)
        p = size(train, 2) - 1
        X = vcat(train[:, 1:p], val[:, 1:p])
        Y = Int.(vcat(train[:, p+1], val[:, p+1]))
        X_test, Y_test = test[:, 1:p], Int.(test[:, p+1])
        classes = sort(unique(Y))            # get a dataframe with the classes in the data
        class_labels = size(classes, 1)      # number of classes  
        tmp_Y = zeros(Int, size(Y, 1))
        tmp_Y_test = zeros(Int, size(Y_test, 1))
        for i in 1:class_labels
            tmp_Y[findall(x -> x == classes[i], Y)] .= i
            tmp_Y_test[findall(x -> x == classes[i], Y_test)] .= i
        end
        Y = tmp_Y
        Y_test = tmp_Y_test
        classes = sort(unique(Y))  
        class_labels = size(unique(Y), 1)   # number of classes  
        println("dataset: ", dataset, " run: ", run, " n_train: ", size(X, 1), " n: ", size(vcat(X, X_test), 1), " p: ", p, " class_labels: ", size(unique(Y), 1))
            
        alpha_best = 0
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
            C, train = MHABR.Utils.route(X, Y, A, B)
            test = MHABR.Utils.eva(X_test, Y_test, A, B, C)
            splits = count(!iszero, A)
            println(" Alpha: ", alpha_i, " number of splits: ", splits)
            number_splits[index] = splits
            if test < opt_test
                alpha_best = alpha_i
                opt_train = train
                opt_test = test
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
        println("Train Acc: ", train_acc, " Train_cost ", opt_train, " Test Acc: ", test_acc, " Time: ", end_time - start," Splits: ", num_splits)
        result_train[dataset, column] = train_acc
        result_test[dataset, column] = test_acc
        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_accuracys_updated.csv", DataFrame(train_accs, :auto))
        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_test_accuracys_updated.csv", DataFrame(test_accs, :auto))
        CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_times_updated.csv", DataFrame(train_times, :auto))
    end # run
end # dataset
# std and mean of the results
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
#println("Splits_number: ", split_num[:, len+1])
result_train[:, len+1] = mean(result_train[:, 1:len], dims=2)
result_test[:, len+1] = mean(result_test[:, 1:len], dims=2)
CSV.write("Train_Result", DataFrame(result_train, :auto))
CSV.write("Train_Test", DataFrame(result_test, :auto))
# save the results
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_accuracys_updated.csv", DataFrame(train_accs, :auto))
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_test_accuracys_updated.csv", DataFrame(test_accs, :auto))
CSV.write("CART_MH_" * string(dataset_start) * "_" * string(dataset_end) * "_time_limit_" * string(time_limit) * "_runs_" * string(run_start) * "_" * string(run_end) * "_d_" * string(tree_depth) * "_" * string(epsilon) * "_" * "_Nmin_1_training_times_updated.csv", DataFrame(train_times, :auto))

return 0
end # main

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
