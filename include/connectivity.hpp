#ifndef CONNECTIVITY_HPP
#define CONNECTIVITY_HPP

#include "common_types.hpp"

#include <vector>

namespace snn
{
    struct SynapticConnection
    {
        NeuronId source;
        NeuronId target;
        float weight;
    };

    struct CSRConnectivity
    {
        std::vector<std::size_t> row_offsets;
        std::vector<NeuronId> column_indices;
        std::vector<float> weights;

        std::size_t neuron_count() const;
        std::size_t connection_count() const;
    };

    struct CSCConnectivity
    {
        std::vector<std::size_t> column_offsets;
        std::vector<NeuronId> row_indices;
        std::vector<std::size_t> csc_to_csr;

        std::size_t neuron_count() const;
        std::size_t connection_count() const;
    };

    CSRConnectivity buildCSR(
        std::size_t neuron_count,
        const std::vector<SynapticConnection>& connections
    );

    CSCConnectivity buildCSC(
        std::size_t neuron_count,
        const std::vector<SynapticConnection>& connections
    );

    void processActiveSpikes(
        const CSRConnectivity& connectivity,
        const std::vector<SpikeEvent>& active_spikes,
        std::vector<float>& synaptic_currents
    );
}

#endif