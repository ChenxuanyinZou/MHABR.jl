import CSV
import DataFrames

const _IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

function _load_iris(path::AbstractString)
    isfile(path) || error("Iris CSV not found at $path")
    data = CSV.read(path, DataFrames.DataFrame; header=false)
    size(data) == (150, 5) || error("expected a 150-by-5 Iris CSV at $path")

    X = Matrix{Float64}(data[:, 1:4])
    y = String.(data[:, 5])
    classes = sort(unique(y))
    classes == _IRIS_CLASSES || error("unexpected Iris classes: $classes")
    all(label -> count(==(label), y) == 50, _IRIS_CLASSES) ||
        error("expected 50 observations for each Iris class")
    return X, y
end

"""
    load_iris()

Load the bundled Iris dataset and return `(X, y)`, where `X` is a `150 × 4`
`Matrix{Float64}` and `y` is a 150-element `Vector{String}`.
"""
function load_iris()
    path = normpath(joinpath(@__DIR__, "..", "examples", "data", "iris.csv"))
    return _load_iris(path)
end
