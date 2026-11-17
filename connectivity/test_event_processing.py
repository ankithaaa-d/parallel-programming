import argparse
import numpy as np

from shared_connectivity import SharedConnectivity
from event_processing import (
    collect_synaptic_events,
    PRESYNAPTIC_EVENT,
    POSTSYNAPTIC_EVENT
)


def generate_network(num_neurons, num_synapses, seed=None):
    """Generate a network with unique source-destination pairs."""

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

    return sources, destinations, weights, rng


def generate_spikes(num_neurons, sources, destinations, rng):
    """Generate spike queues that guarantee at least one shared event."""

    pre_spikes = np.unique(
        np.concatenate((
            sources[:1],
            rng.choice(
                num_neurons,
                size=max(1, num_neurons // 4),
                replace=False
            )
        ))
    )

    post_spikes = np.unique(
        np.concatenate((
            destinations[:1],
            rng.choice(
                num_neurons,
                size=max(1, num_neurons // 4),
                replace=False
            )
        ))
    )

    return pre_spikes, post_spikes


def expected_events(network, pre_spikes, post_spikes):
    """Build expected events from the network's CSR and CSC traversals."""

    expected = set()

    # Expected presynaptic events.
    for source in pre_spikes:

        destinations, synapse_ids = (
            network.outgoing_connections(int(source))
        )

        for destination, synapse_id in zip(
            destinations, synapse_ids
        ):

            expected.add((
                int(source),
                int(destination),
                int(synapse_id),
                PRESYNAPTIC_EVENT
            ))

    # Expected postsynaptic events.
    for destination in post_spikes:

        sources, synapse_ids = (
            network.incoming_connections(int(destination))
        )

        for source, synapse_id in zip(sources, synapse_ids):

            expected.add((
                int(source),
                int(destination),
                int(synapse_id),
                POSTSYNAPTIC_EVENT
            ))

    return expected


def actual_events(events):
    """Convert structured event output into comparable tuples."""

    return {
        (
            int(event["source"]),
            int(event["destination"]),
            int(event["synapse_id"]),
            int(event["event_type"])
        )
        for event in events
    }


def test_event_processing(num_neurons, num_synapses, seed=None):

    sources, destinations, weights, rng = generate_network(
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

    pre_spikes, post_spikes = generate_spikes(
        num_neurons,
        sources,
        destinations,
        rng
    )

    # Test normal event processing.
    events = collect_synaptic_events(
        network,
        pre_spikes=pre_spikes,
        post_spikes=post_spikes
    )

    expected = expected_events(
        network,
        pre_spikes,
        post_spikes
    )

    assert actual_events(events) == expected
    assert len(events) == len(expected)

    print("Normal event processing: PASSED")

    # Test simultaneous events on shared synapses.
    pre_events = events[
        events["event_type"] == PRESYNAPTIC_EVENT
    ]

    post_events = events[
        events["event_type"] == POSTSYNAPTIC_EVENT
    ]

    pre_synapses = set(pre_events["synapse_id"].tolist())
    post_synapses = set(post_events["synapse_id"].tolist())

    shared_synapses = pre_synapses.intersection(post_synapses)

    assert len(shared_synapses) > 0

    for synapse_id in shared_synapses:

        matching_events = events[
            events["synapse_id"] == synapse_id
        ]

        event_types = set(
            matching_events["event_type"].tolist()
        )

        assert event_types == {
            PRESYNAPTIC_EVENT,
            POSTSYNAPTIC_EVENT
        }

    print("Simultaneous event verification: PASSED")

    # Test empty spike queues.
    empty_events = collect_synaptic_events(
        network,
        pre_spikes=[],
        post_spikes=[]
    )

    assert len(empty_events) == 0

    print("Empty spike queue: PASSED")

    # Test presynaptic-only processing.
    pre_only = collect_synaptic_events(
        network,
        pre_spikes=pre_spikes,
        post_spikes=[]
    )

    expected_pre = expected_events(
        network,
        pre_spikes,
        []
    )

    assert actual_events(pre_only) == expected_pre
    assert np.all(
        pre_only["event_type"] == PRESYNAPTIC_EVENT
    )

    print("Presynaptic-only processing: PASSED")

    # Test postsynaptic-only processing.
    post_only = collect_synaptic_events(
        network,
        pre_spikes=[],
        post_spikes=post_spikes
    )

    expected_post = expected_events(
        network,
        [],
        post_spikes
    )

    assert actual_events(post_only) == expected_post
    assert np.all(
        post_only["event_type"] == POSTSYNAPTIC_EVENT
    )

    print("Postsynaptic-only processing: PASSED")

    print("\nNetwork neurons:", num_neurons)
    print("Network synapses:", num_synapses)
    print("Presynaptic spikes:", len(pre_spikes))
    print("Postsynaptic spikes:", len(post_spikes))
    print("Generated events:", len(events))

    print("\nALL EVENT PROCESSING TESTS PASSED")


def main():

    parser = argparse.ArgumentParser(
        description="Dynamic synaptic event-processing verification"
    )

    parser.add_argument(
        "--num-neurons",
        type=int,
        required=True,
        help="Number of neurons"
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

    test_event_processing(
        args.num_neurons,
        args.num_synapses,
        args.seed
    )


if __name__ == "__main__":
    main()