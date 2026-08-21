using CSV
using DataFrames
using MHABR
using Random

const IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
const FEATURE_NAMES = ["sepal length", "sepal width", "petal length", "petal width"]

function load_iris(path=joinpath(@__DIR__, "data", "iris.csv"))
    data = CSV.read(path, DataFrame; header=false)
    size(data) == (150, 5) || error("expected a 150-by-5 Iris CSV at $path")

    X = Matrix{Float64}(data[:, 1:4])
    y = String.(data[:, 5])
    classes = sort(unique(y))
    classes == IRIS_CLASSES || error("unexpected Iris classes: $classes")
    all(label -> count(==(label), y) == 50, IRIS_CLASSES) ||
        error("expected 50 observations for each Iris class")
    return X, y
end

function iris_split()
    train = vcat(1:40, 51:90, 101:140)
    test = vcat(41:50, 91:100, 141:150)
    return train, test
end

function main()
    X, y = load_iris()
    train, test = iris_split()

    tree = fit(
        X[train, :],
        y[train];
        max_depth=2,
        epsilon=0.0,
        alpha=0.0,
        batch=1.0,
        rng=MersenneTwister(1),
        time_limit=10.0,
    )

    predictions = predict(tree, X[test, :])
    any(ismissing, predictions) && error("Iris example produced missing predictions")
    accuracy = score(tree, X[test, :], y[test])
    accuracy >= 0.80 || error("Iris test accuracy $accuracy is below 0.80")

    println("Iris observations: ", size(X, 1))
    println("Features: ", FEATURE_NAMES)
    println("Classes: ", IRIS_CLASSES)
    println("Training/test observations: ", length(train), "/", length(test))
    println(DataFrame(actual=y[test], predicted=coalesce.(predictions, "missing")))
    println("Test accuracy: ", accuracy)
    println("Tree (feature indices follow the feature list above):")
    print_tree(tree)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
