import argparse
import numpy as np

from shared_connectivity import SharedConnectivity


def generate_network(num_neurons, num_synapses, seed=None):
    """Generate connectivity using runtime-provided network dimensions."""

    if num_neurons <= 0:
        raise ValueError("Number of neurons must be positive.")

    if num_synapses <= 0:
        raise ValueError("Number of synapses must be positive.")

    if num_synapses > num_neurons * num_neurons:
        raise ValueError(
            "Cannot generate more unique connections than "
            "the total possible neuron pairs."
        )

    rng = np.random.default_rng(seed)

    edge_ids = rng.choice(
        num_neurons * num_neurons,
        size=num_synapses,
        replace=False
    )

    sources = edge_ids // num_neurons
    destinations = edge_ids % num_neurons
    weights = rng.uniform(0.01, 1.0, size=num_synapses)

    return sources, destinations, weights


def test_network(num_neurons, num_synapses, seed=None):

    sources, destinations, weights = generate_network(
        num_neurons,
        num_synapses,
        seed
    )

    network = SharedConnectivity(
        num_neurons=num_neurons,
        sources=sources,
        destinations=destinations,
        weights=weights
    )

    expected_edges = {
        (int(source), int(destination)): float(weight)
        for source, destination, weight
        in zip(sources, destinations, weights)
    }

    csr_edges = {}
    csc_edges = {}

    # Verify every CSR connection.
    for neuron in range(num_neurons):

        destination_ids, synapse_ids = (
            network.outgoing_connections(neuron)
        )

        for destination, synapse_id in zip(
            destination_ids, synapse_ids
        ):

            edge = (neuron, int(destination))

            assert edge not in csr_edges

            csr_edges[edge] = network.weights[synapse_id]

    assert set(csr_edges) == set(expected_edges)

    print("CSR connection verification: PASSED")

    # Verify every CSC connection.
    for neuron in range(num_neurons):

        source_ids, synapse_ids = (
            network.incoming_connections(neuron)
        )

        for source, synapse_id in zip(source_ids, synapse_ids):

            edge = (int(source), neuron)

            assert edge not in csc_edges

            csc_edges[edge] = network.weights[synapse_id]

    assert set(csc_edges) == set(expected_edges)

    print("CSC connection verification: PASSED")

    # Verify shared weights.
    for edge, expected_weight in expected_edges.items():

        assert np.isclose(csr_edges[edge], expected_weight)
        assert np.isclose(csc_edges[edge], expected_weight)

    print("Shared weight verification: PASSED")

    # Verify CSC-to-CSR mapping.
    assert len(network.csc_to_csr) == num_synapses

    assert np.array_equal(
        np.sort(network.csc_to_csr),
        np.arange(num_synapses)
    )

    for destination in range(num_neurons):

        start = network.csc_indptr[destination]
        end = network.csc_indptr[destination + 1]

        for position in range(start, end):

            source = int(network.csc_indices[position])
            synapse_id = int(network.csc_to_csr[position])

            # Verify mapped synapse belongs to the correct source.
            assert (
                network.row_ptr[source]
                <= synapse_id
                < network.row_ptr[source + 1]
            )

            # Verify mapped synapse reaches the correct destination.
            assert network.col_idx[synapse_id] == destination

            # Verify the mapped weight matches the original connection.
            assert np.isclose(
                network.weights[synapse_id],
                expected_edges[(source, destination)]
            )

    print("CSC-to-CSR weight mapping verification: PASSED")

    # Verify shared weight update.
    rng = np.random.default_rng()

    synapse_id = int(rng.integers(0, num_synapses))
    new_weight = float(rng.uniform(0.01, 1.0))

    network.update_weight(synapse_id, new_weight)

    assert np.isclose(
        network.weights[synapse_id],
        new_weight
    )

    print("Shared weight update: PASSED")

    # Verify weight retrieval.
    retrieved_weights = network.get_weights(
        np.arange(num_synapses)
    )

    assert np.allclose(
        retrieved_weights,
        network.weights
    )

    print("Weight retrieval: PASSED")

    print(
        f"Neurons: {num_neurons}, "
        f"Synapses: {num_synapses} - PASSED"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Dynamic CSR and CSC connectivity verification"
    )

    parser.add_argument(
        "--num-neurons",
        type=int,
        required=True,
        help="Number of neurons in the network"
    )

    parser.add_argument(
        "--num-synapses",
        type=int,
        required=True,
        help="Number of unique synaptic connections"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility"
    )

    args = parser.parse_args()

    test_network(
        args.num_neurons,
        args.num_synapses,
        args.seed
    )

    print("\nALL DYNAMIC CONNECTIVITY TESTS PASSED")


if __name__ == "__main__":
    main()