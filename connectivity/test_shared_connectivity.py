import numpy as np
from shared_connectivity import SharedConnectivity


def generate_test_connectivity():
    rng = np.random.default_rng()

    num_neurons = int(rng.integers(10, 101))
    num_synapses = int(rng.integers(num_neurons, num_neurons * 10))

    sources = rng.integers(0, num_neurons, size=num_synapses)
    destinations = rng.integers(0, num_neurons, size=num_synapses)
    weights = rng.uniform(0.01, 1.0, size=num_synapses)

    return num_neurons, sources, destinations, weights


def test_shared_connectivity():
    num_neurons, sources, destinations, weights = (
        generate_test_connectivity()
    )

    network = SharedConnectivity(
        num_neurons=num_neurons,
        sources=sources,
        destinations=destinations,
        weights=weights
    )

    # CSR structure
    assert len(network.row_ptr) == num_neurons + 1
    assert network.row_ptr[0] == 0
    assert network.row_ptr[-1] == len(network.col_idx)
    assert np.all(np.diff(network.row_ptr) >= 0)

    print("CSR structure: PASSED")

    # CSC structure
    assert len(network.csc_indptr) == num_neurons + 1
    assert network.csc_indptr[0] == 0
    assert network.csc_indptr[-1] == len(network.csc_indices)
    assert np.all(np.diff(network.csc_indptr) >= 0)

    print("CSC structure: PASSED")

    # Shared weight storage and mapping
    assert len(network.weights) == len(network.col_idx)
    assert len(network.csc_to_csr) == len(network.csc_indices)
    assert np.all(network.csc_to_csr >= 0)
    assert np.all(network.csc_to_csr < len(network.weights))

    print("Shared weight mapping: PASSED")

    # Verify incoming connections and CSC-to-CSR mapping
    for destination in range(num_neurons):
        incoming_sources, synapse_ids = (
            network.incoming_connections(destination)
        )

        start = network.csc_indptr[destination]
        end = network.csc_indptr[destination + 1]

        assert np.array_equal(
            incoming_sources,
            network.csc_indices[start:end]
        )

        assert np.array_equal(
            synapse_ids,
            network.csc_to_csr[start:end]
        )

        assert np.all(network.col_idx[synapse_ids] == destination)

    print("Incoming traversal and CSC-to-CSR mapping: PASSED")

    # Verify outgoing connections
    for source in range(num_neurons):
        outgoing_destinations, synapse_ids = (
            network.outgoing_connections(source)
        )

        start = network.row_ptr[source]
        end = network.row_ptr[source + 1]

        assert np.array_equal(
            outgoing_destinations,
            network.col_idx[start:end]
        )

        assert np.array_equal(
            synapse_ids,
            np.arange(start, end)
        )

    print("Outgoing traversal: PASSED")

    # Verify shared weight update
    if len(network.weights) > 0:
        rng = np.random.default_rng()
        synapse_id = int(rng.integers(0, len(network.weights)))
        new_weight = float(rng.uniform(0.01, 1.0))

        network.update_weight(synapse_id, new_weight)

        assert np.isclose(network.weights[synapse_id], new_weight)

        print("Shared weight update: PASSED")

    # Verify weight retrieval
    retrieved_weights = network.get_weights(
        np.arange(len(network.weights))
    )

    assert np.allclose(retrieved_weights, network.weights)

    print("Weight retrieval: PASSED")
    print("ALL SHARED CONNECTIVITY TESTS PASSED")


if __name__ == "__main__":
    test_shared_connectivity()