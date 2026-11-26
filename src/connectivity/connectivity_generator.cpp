#include "connectivity.hpp"

#include <cmath>
#include <stdexcept>

namespace snn
{
    void processActiveSpikes(
        const CSRConnectivity& connectivity,
        const std::vector<SpikeEvent>& active_spikes,
        std::vector<float>& synaptic_currents)
    {
        const std::size_t neurons = connectivity.neuron_count();

        // Validate connectivity structure.
        if (connectivity.row_offsets.size() != neurons + 1 ||
            connectivity.column_indices.size() != connectivity.weights.size())
        {
            throw std::invalid_argument(
                "Invalid CSR connectivity structure.");
        }

        if (synaptic_currents.size() != neurons)
        {
            throw std::invalid_argument(
                "Synaptic current array has incorrect size.");
        }

        if (connectivity.row_offsets.front() != 0 ||
            connectivity.row_offsets.back() !=
                connectivity.column_indices.size())
        {
            throw std::invalid_argument(
                "Invalid CSR row offsets.");
        }

        // Process every active presynaptic spike.
        for (const auto& spike : active_spikes)
        {
            const NeuronId source = spike.neuron_id;

            if (source >= neurons)
            {
                throw std::out_of_range(
                    "Active spike contains invalid neuron ID.");
            }

            const std::size_t start =
                connectivity.row_offsets[source];

            const std::size_t end =
                connectivity.row_offsets[source + 1];

            if (start > end || end > connectivity.weights.size())
            {
                throw std::invalid_argument(
                    "Invalid CSR row boundaries.");
            }

            // Traverse outgoing synaptic connections.
            for (std::size_t edge = start; edge < end; ++edge)
            {
                const NeuronId target =
                    connectivity.column_indices[edge];

                const float weight = connectivity.weights[edge];

                if (target >= neurons || !std::isfinite(weight))
                {
                    throw std::invalid_argument(
                        "Invalid synaptic connection.");
                }

                // Accumulate synaptic input at the target neuron.
                synaptic_currents[target] += weight;
            }
        }
    }
}