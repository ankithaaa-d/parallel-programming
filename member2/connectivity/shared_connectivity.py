import numpy as np


class SharedConnectivity:
    """
    Dynamic CSR + CSC connectivity structure.

    All network information is provided through input arrays.

    CSR:
        row_ptr
        col_idx
        weights

    CSC:
        csc_indptr
        csc_indices
        csc_to_csr

    Synaptic weights are stored only once, in CSR order.
    """

    def __init__(
        self,
        num_neurons,
        sources,
        destinations,
        weights
    ):

        if (
            isinstance(num_neurons, (bool, np.bool_))
            or not isinstance(num_neurons, (int, np.integer))
            or num_neurons <= 0
        ):
            raise ValueError(
                "num_neurons must be a positive integer"
            )

        self.num_neurons = int(num_neurons)

        sources = self._validate_integer_array(
            sources, "sources"
        )

        destinations = self._validate_integer_array(
            destinations, "destinations"
        )

        weights = np.asarray(weights)

        if weights.ndim != 1:
            raise ValueError("Weights must be one-dimensional")

        if weights.dtype.kind not in "iuf":
            raise TypeError("Weights must be numeric")

        weights = weights.astype(np.float64, copy=False)

        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must be finite")

        if not (
            len(sources)
            == len(destinations)
            == len(weights)
        ):
            raise ValueError(
                "Sources, destinations and weights "
                "must have equal lengths"
            )

        if np.any(sources < 0) or np.any(
            sources >= self.num_neurons
        ):
            raise ValueError("Invalid source neuron ID")

        if np.any(destinations < 0) or np.any(
            destinations >= self.num_neurons
        ):
            raise ValueError("Invalid destination neuron ID")

        self.num_synapses = len(weights)

        self._build_csr(
            sources,
            destinations,
            weights
        )

        self._build_csc(
            sources,
            destinations
        )

    @staticmethod
    def _validate_integer_array(values, name):

        array = np.asarray(values)

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional"
            )

        if array.size == 0:
            return np.empty(0, dtype=np.int64)

        if array.dtype.kind not in "iu":
            raise TypeError(
                f"{name} must contain integer neuron IDs"
            )

        return array.astype(np.int64, copy=False)

    def _build_csr(
        self,
        sources,
        destinations,
        weights
    ):

        # Sort outgoing connections by source and destination.
        csr_order = np.lexsort(
            (destinations, sources)
        )

        csr_sources = sources[csr_order]

        self.col_idx = destinations[csr_order].copy()

        # The only physical weight array.
        self.weights = weights[csr_order].copy()

        connection_counts = np.bincount(
            csr_sources,
            minlength=self.num_neurons
        )

        self.row_ptr = np.zeros(
            self.num_neurons + 1,
            dtype=np.int64
        )

        self.row_ptr[1:] = np.cumsum(
            connection_counts
        )

    def _build_csc(
        self,
        sources,
        destinations
    ):

        # Sort incoming connections by destination and source.
        csc_order = np.lexsort(
            (sources, destinations)
        )

        self.csc_indices = sources[csc_order].copy()

        connection_counts = np.bincount(
            destinations[csc_order],
            minlength=self.num_neurons
        )

        self.csc_indptr = np.zeros(
            self.num_neurons + 1,
            dtype=np.int64
        )

        self.csc_indptr[1:] = np.cumsum(
            connection_counts
        )

        # Map original edge positions to CSR positions.
        csr_order = np.lexsort(
            (destinations, sources)
        )

        csr_positions = np.empty(
            self.num_synapses,
            dtype=np.int64
        )

        csr_positions[csr_order] = np.arange(
            self.num_synapses
        )

        # Every CSC entry points to its shared CSR weight.
        self.csc_to_csr = csr_positions[csc_order]

    def _validate_neuron(self, neuron_id):

        if (
            isinstance(neuron_id, (bool, np.bool_))
            or not isinstance(neuron_id, (int, np.integer))
        ):
            raise TypeError(
                "Neuron ID must be an integer"
            )

        if neuron_id < 0 or neuron_id >= self.num_neurons:
            raise ValueError(
                "Neuron ID is outside the valid range"
            )

    def outgoing_connections(self, neuron_id):
        """
        Retrieve outgoing connections using CSR.

        Returns:
            destination neuron IDs
            shared synapse IDs
        """

        self._validate_neuron(neuron_id)

        start = self.row_ptr[neuron_id]
        end = self.row_ptr[neuron_id + 1]

        destinations = self.col_idx[start:end]

        synapse_ids = np.arange(
            start,
            end,
            dtype=np.int64
        )

        return destinations, synapse_ids

    def incoming_connections(self, neuron_id):
        """
        Retrieve incoming connections using CSC.

        Returns:
            source neuron IDs
            shared synapse IDs
        """

        self._validate_neuron(neuron_id)

        start = self.csc_indptr[neuron_id]
        end = self.csc_indptr[neuron_id + 1]

        sources = self.csc_indices[start:end]

        synapse_ids = self.csc_to_csr[start:end]

        return sources, synapse_ids

    def get_weights(self, synapse_ids):

        synapse_ids = self._validate_integer_array(
            synapse_ids,
            "synapse_ids"
        )

        if np.any(synapse_ids < 0) or np.any(
            synapse_ids >= self.num_synapses
        ):
            raise ValueError("Invalid synapse ID")

        return self.weights[synapse_ids]

    def update_weight(self, synapse_id, new_weight):

        if (
            isinstance(synapse_id, (bool, np.bool_))
            or not isinstance(synapse_id, (int, np.integer))
        ):
            raise TypeError(
                "Synapse ID must be an integer"
            )

        if synapse_id < 0 or synapse_id >= self.num_synapses:
            raise ValueError("Invalid synapse ID")

        if (
            isinstance(new_weight, (bool, np.bool_))
            or not np.isscalar(new_weight)
            or not np.isfinite(new_weight)
        ):
            raise ValueError("New weight must be finite")

        self.weights[synapse_id] = new_weight