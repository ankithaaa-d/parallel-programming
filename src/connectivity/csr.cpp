#include "connectivity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace snn
{
    std::size_t CSRConnectivity::neuron_count() const
    {
        return row_offsets.empty() ? 0 : row_offsets.size() - 1;
    }

    std::size_t CSRConnectivity::connection_count() const
    {
        return column_indices.size();
    }

    CSRConnectivity buildCSR(
        std::size_t neuron_count,
        const std::vector<SynapticConnection>& connections)
    {
        CSRConnectivity csr;

        csr.row_offsets.assign(neuron_count + 1, 0);

        std::vector<SynapticConnection> sorted = connections;

        // Validate connections and count outgoing edges.
        for (const auto& connection : sorted)
        {
            if (connection.source >= neuron_count ||
                connection.target >= neuron_count)
            {
                throw std::invalid_argument(
                    "Connection contains an invalid neuron ID.");
            }

            if (!std::isfinite(connection.weight))
            {
                throw std::invalid_argument(
                    "Synaptic weight must be finite.");
            }

            ++csr.row_offsets[connection.source + 1];
        }

        // Convert counts into row offsets.
        for (std::size_t i = 1; i <= neuron_count; ++i)
        {
            csr.row_offsets[i] += csr.row_offsets[i - 1];
        }

        // Group connections by source neuron.
        std::stable_sort(
            sorted.begin(),
            sorted.end(),
            [](const SynapticConnection& lhs,
               const SynapticConnection& rhs)
            {
                return lhs.source < rhs.source;
            });

        // Store destination neurons and corresponding weights.
        csr.column_indices.reserve(sorted.size());
        csr.weights.reserve(sorted.size());

        for (const auto& connection : sorted)
        {
            csr.column_indices.push_back(connection.target);
            csr.weights.push_back(connection.weight);
        }

        return csr;
    }
}