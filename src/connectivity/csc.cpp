#include "connectivity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace snn
{
    std::size_t CSCConnectivity::neuron_count() const
    {
        return column_offsets.empty() ? 0 : column_offsets.size() - 1;
    }

    std::size_t CSCConnectivity::connection_count() const
    {
        return row_indices.size();
    }

    CSCConnectivity buildCSC(
        std::size_t neuron_count,
        const std::vector<SynapticConnection>& connections)
    {
        CSCConnectivity csc;

        csc.column_offsets.assign(neuron_count + 1, 0);

        std::vector<std::size_t> sorted_indices(connections.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        for (const auto& connection : connections)
        {
            if (connection.source >= neuron_count ||
                connection.target >= neuron_count)
                throw std::invalid_argument("Invalid neuron ID.");

            if (!std::isfinite(connection.weight))
                throw std::invalid_argument("Invalid synaptic weight.");

            ++csc.column_offsets[connection.target + 1];
        }

        for (std::size_t i = 1; i <= neuron_count; ++i)
            csc.column_offsets[i] += csc.column_offsets[i - 1];

        std::stable_sort(
            sorted_indices.begin(),
            sorted_indices.end(),
            [&](std::size_t a, std::size_t b)
            {
                return connections[a].target <
                       connections[b].target;
            });

        csc.row_indices.reserve(connections.size());
        csc.csc_to_csr.reserve(connections.size());

        // CSR order is the original input order after sorting by source.
        std::vector<std::size_t> csr_order(connections.size());
        std::iota(csr_order.begin(), csr_order.end(), 0);

        std::stable_sort(
            csr_order.begin(),
            csr_order.end(),
            [&](std::size_t a, std::size_t b)
            {
                return connections[a].source <
                       connections[b].source;
            });

        // Map each original connection to its CSR position.
        std::vector<std::size_t> original_to_csr(connections.size());

        for (std::size_t i = 0; i < csr_order.size(); ++i)
            original_to_csr[csr_order[i]] = i;

        for (const auto original_index : sorted_indices)
        {
            csc.row_indices.push_back(
                connections[original_index].source);

            csc.csc_to_csr.push_back(
                original_to_csr[original_index]);
        }

        return csc;
    }
}