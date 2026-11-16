import argparse
import numpy as np

from shared_connectivity import SharedConnectivity
from event_processing import collect_synaptic_events


def generate_network(num_neurons, num_synapses, seed=None):
    """Generate a network using runtime-provided dimensions."""

    if num_neurons <= 0:
        raise ValueError("Number of neurons must be positive.")

    if num_synapses <= 0:
        raise ValueError("Number of synapses must be positive.")

    if num_synapses > num_neurons * num_neurons:
        raise ValueError("Too many synapses for the network size.")

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


def expect_exception(function, expected_exception, test_name):
    """Verify that an operation raises the expected exception."""

    try:
        function()
    except expected_exception:
        print(f"{test_name}: PASSED")
        return

    raise AssertionError(
        f"{test_name}: Expected {expected_exception.__name__} "
        "but no matching exception was raised."
    )


def test_invalid_inputs(num_neurons, num_synapses, seed=None):

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

    # Test neuron ID outside the network range.
    expect_exception(
        lambda: collect_synaptic_events(
            network,
            [num_neurons],
            []
        ),
        ValueError,
        "Invalid neuron ID"
    )

    # Test negative neuron ID.
    expect_exception(
        lambda: collect_synaptic_events(
            network,
            [-1],
            []
        ),
        ValueError,
        "Negative neuron ID"
    )

    # Test duplicate spike IDs.
    valid_neuron = int(sources[0])

    expect_exception(
        lambda: collect_synaptic_events(
            network,
            [valid_neuron, valid_neuron],
            []
        ),
        ValueError,
        "Duplicate spike validation"
    )

    # Test non-integer neuron IDs.
    expect_exception(
        lambda: collect_synaptic_events(
            network,
            [1.5],
            []
        ),
        TypeError,
        "Non-integer spike validation"
    )

    print("\nNetwork neurons:", num_neurons)
    print("Network synapses:", num_synapses)
    print("\nALL EVENT VALIDATION TESTS PASSED")


def main():

    parser = argparse.ArgumentParser(
        description="Dynamic event input validation tests"
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

    test_invalid_inputs(
        args.num_neurons,
        args.num_synapses,
        args.seed
    )


if __name__ == "__main__":
    main()