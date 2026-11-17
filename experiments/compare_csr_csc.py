import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix


def load_connectivity(file_path, num_neurons=None):
    """Load network connectivity from a CSV file."""

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

    return sources, destinations, weights, num_neurons


def calculate_memory(sparse_matrix):
    """Calculate memory used by a compressed sparse matrix."""

    return (
        sparse_matrix.data.nbytes
        + sparse_matrix.indices.nbytes
        + sparse_matrix.indptr.nbytes
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic CSR and CSC comparison"
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

    sources, destinations, weights, num_neurons = load_connectivity(
        args.input,
        args.num_neurons
    )

    matrix_shape = (num_neurons, num_neurons)

    # Construct CSR and measure conversion time.
    start_time = time.perf_counter()

    csr = csr_matrix(
        (weights, (sources, destinations)),
        shape=matrix_shape
    )

    csr_time = time.perf_counter() - start_time

    # Construct CSC and measure conversion time.
    start_time = time.perf_counter()

    csc = csc_matrix(
        (weights, (sources, destinations)),
        shape=matrix_shape
    )

    csc_time = time.perf_counter() - start_time

    # Verify both representations against the input connectivity.
    expected = np.zeros(matrix_shape, dtype=np.float64)
    np.add.at(expected, (sources, destinations), weights)

    csr_correct = np.allclose(csr.toarray(), expected)
    csc_correct = np.allclose(csc.toarray(), expected)
    equivalent = np.allclose(csr.toarray(), csc.toarray())

    if not csr_correct or not csc_correct or not equivalent:
        raise AssertionError("CSR/CSC verification failed.")

    # Calculate memory consumption.
    csr_memory = calculate_memory(csr)
    csc_memory = calculate_memory(csc)

    # Display results.
    print("CSR AND CSC COMPARISON")
    print("----------------------")

    print("Number of neurons:", num_neurons)
    print("Input connections:", len(weights))

    print("\nCorrectness Verification")
    print("CSR correct:", csr_correct)
    print("CSC correct:", csc_correct)
    print("CSR and CSC equivalent:", equivalent)

    print("\nNon-zero Entries")
    print("CSR:", csr.nnz)
    print("CSC:", csc.nnz)

    print("\nMemory Comparison")
    print("CSR:", csr_memory, "bytes")
    print("CSC:", csc_memory, "bytes")

    print("\nConstruction Time")
    print("CSR:", csr_time, "seconds")
    print("CSC:", csc_time, "seconds")


if __name__ == "__main__":
    main()