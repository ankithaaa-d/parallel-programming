import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_connectivity(file_path, num_neurons=None):

    # Read connectivity from an external CSV file.
    data = pd.read_csv(file_path)

    required_columns = {"source", "destination", "weight"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain source, destination and weight columns"
        )

    sources = data["source"].to_numpy()
    destinations = data["destination"].to_numpy()
    weights = data["weight"].to_numpy(dtype=np.float64)

    if len(sources) == 0:
        raise ValueError("Connectivity file contains no connections")

    if not np.issubdtype(sources.dtype, np.integer):
        raise TypeError("Source IDs must be integers")

    if not np.issubdtype(destinations.dtype, np.integer):
        raise TypeError("Destination IDs must be integers")

    if np.any(sources < 0) or np.any(destinations < 0):
        raise ValueError("Neuron IDs cannot be negative")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite")

    inferred_neurons = int(
        max(sources.max(), destinations.max()) + 1
    )

    if num_neurons is None:
        num_neurons = inferred_neurons

    if num_neurons < inferred_neurons:
        raise ValueError(
            "Neuron count is smaller than the maximum neuron ID"
        )

    return sources, destinations, weights, num_neurons


def run_csr_experiment(file_path, num_neurons=None):

    sources, destinations, weights, num_neurons = (
        load_connectivity(file_path, num_neurons)
    )

    # Construct CSR directly from connectivity data.
    csr = csr_matrix(
        (weights, (sources, destinations)),
        shape=(num_neurons, num_neurons)
    )

    print("\nCSR CONNECTIVITY ANALYSIS")

    print("\nNumber of neurons:", num_neurons)
    print("Number of input connections:", len(weights))
    print("CSR nonzero entries:", csr.nnz)

    print("\nCSR Data:", csr.data)
    print("CSR Indices:", csr.indices)
    print("CSR Indptr:", csr.indptr)

    # Reconstruct matrix for verification.
    reconstructed = csr.toarray()

    print("\nReconstructed Connectivity Matrix:")
    print(reconstructed)

    # Verify every input connection.
    expected = np.zeros(
        (num_neurons, num_neurons),
        dtype=np.float64
    )

    np.add.at(expected, (sources, destinations), weights)

    verified = np.allclose(reconstructed, expected)

    print("\nConnectivity Verification:", verified)

    # Memory analysis.
    dense_memory = expected.nbytes

    csr_memory = (
        csr.data.nbytes
        + csr.indices.nbytes
        + csr.indptr.nbytes
    )

    print("\nMEMORY ANALYSIS")
    print("Dense memory:", dense_memory, "bytes")
    print("CSR memory:", csr_memory, "bytes")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Dynamic CSR connectivity analysis"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to connectivity CSV file"
    )

    parser.add_argument(
        "--num-neurons",
        type=int,
        default=None,
        help="Total neuron count, including isolated neurons"
    )

    args = parser.parse_args()

    run_csr_experiment(
        args.input,
        args.num_neurons
    )