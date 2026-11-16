import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


def load_connectivity(file_path, num_neurons=None):
    """
    Load connectivity from a CSV file and construct a CSC matrix.

    Required CSV columns:
        source, destination, weight
    """

    data = pd.read_csv(file_path)

    required_columns = {"source", "destination", "weight"}

    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(
            f"CSV file is missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("The connectivity CSV file is empty.")

    sources = data["source"].to_numpy()
    destinations = data["destination"].to_numpy()
    weights = data["weight"].to_numpy(dtype=np.float64)

    if not np.issubdtype(sources.dtype, np.integer):
        raise ValueError("Source neuron IDs must be integers.")

    if not np.issubdtype(destinations.dtype, np.integer):
        raise ValueError("Destination neuron IDs must be integers.")

    if np.any(sources < 0) or np.any(destinations < 0):
        raise ValueError("Neuron IDs cannot be negative.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Synaptic weights must be finite numbers.")

    if num_neurons is None:
        num_neurons = int(max(sources.max(), destinations.max())) + 1

    if not isinstance(num_neurons, int) or num_neurons <= 0:
        raise ValueError("Number of neurons must be a positive integer.")

    if np.any(sources >= num_neurons) or np.any(destinations >= num_neurons):
        raise ValueError(
            "A neuron ID is outside the specified network size."
        )

    connectivity = csc_matrix(
        (weights, (sources, destinations)),
        shape=(num_neurons, num_neurons)
    )

    return connectivity, sources, destinations, weights


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic CSC connectivity analysis"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the connectivity CSV file"
    )

    parser.add_argument(
        "--num-neurons",
        type=int,
        default=None,
        help="Number of neurons (inferred from CSV if omitted)"
    )

    args = parser.parse_args()

    csc, sources, destinations, weights = load_connectivity(
        args.input,
        args.num_neurons
    )

    num_neurons = csc.shape[0]

    print("CSC Connectivity Analysis")
    print("-------------------------")

    print("Number of neurons:", num_neurons)
    print("Number of input connections:", len(weights))
    print("Number of stored CSC entries:", csc.nnz)

    print("\nCSC Arrays")
    print("Data:", csc.data)
    print("Indices:", csc.indices)
    print("Indptr:", csc.indptr)

    reconstructed = csc.toarray()

    print("\nReconstructed Matrix:")
    print(reconstructed)

    expected = np.zeros((num_neurons, num_neurons), dtype=np.float64)
    np.add.at(expected, (sources, destinations), weights)

    print("\nVerification:")
    if np.allclose(reconstructed, expected):
        print("CSC verification: PASSED")
    else:
        raise AssertionError("CSC reconstruction verification failed.")

    dense_memory = expected.nbytes

    csc_memory = (
        csc.data.nbytes
        + csc.indices.nbytes
        + csc.indptr.nbytes
    )

    print("\nMemory Analysis")
    print("Dense memory:", dense_memory, "bytes")
    print("CSC memory:", csc_memory, "bytes")


if __name__ == "__main__":
    main()