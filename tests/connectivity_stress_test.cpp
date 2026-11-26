#include "connectivity.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main()
{
    try
    {
        // Configurable network parameters
        const std::size_t neuron_count = 1000;
        const std::size_t connection_count = 10000;
        const unsigned int seed = 12345;

        std::mt19937 generator(seed);

        std::uniform_int_distribution<std::size_t>
            neuron_distribution(0, neuron_count - 1);

        std::uniform_real_distribution<float>
            weight_distribution(0.1f, 5.0f);

        // Generate random connections
        std::vector<snn::SynapticConnection> connections;
        connections.reserve(connection_count);

        for (std::size_t i = 0; i < connection_count; ++i)
        {
            connections.push_back({
                neuron_distribution(generator),
                neuron_distribution(generator),
                weight_distribution(generator)
            });
        }

        std::cout << "Generated neurons: " << neuron_count << '\n';
        std::cout << "Generated connections: "
                  << connections.size() << '\n';

        // Build CSR and CSC
        const auto csr = snn::buildCSR(neuron_count, connections);
        const auto csc = snn::buildCSC(neuron_count, connections);

        if (csr.connection_count() != connection_count ||
            csc.connection_count() != connection_count ||
            csc.csc_to_csr.size() != connection_count)
            throw std::runtime_error("Connection count mismatch.");

        // Validate CSR and CSC against original connections using shared weights
        std::vector<float> csr_matrix(neuron_count * neuron_count, 0.0f);
        std::vector<float> csc_matrix(neuron_count * neuron_count, 0.0f);
        std::vector<bool> csr_index_visited(connection_count, false);

        for (std::size_t source = 0; source < neuron_count; ++source)
        {
            for (std::size_t edge = csr.row_offsets[source];
                 edge < csr.row_offsets[source + 1]; ++edge)
            {
                const std::size_t target = csr.column_indices[edge];

                csr_matrix[source * neuron_count + target] +=
                    csr.weights[edge];
            }
        }

        for (std::size_t target = 0; target < neuron_count; ++target)
        {
            for (std::size_t edge = csc.column_offsets[target];
                 edge < csc.column_offsets[target + 1]; ++edge)
            {
                const std::size_t source = csc.row_indices[edge];
                const std::size_t csr_edge = csc.csc_to_csr[edge];

                if (csr_edge >= connection_count)
                    throw std::runtime_error("csc_to_csr index out of range.");

                if (csr_index_visited[csr_edge])
                    throw std::runtime_error("Duplicate csc_to_csr mapping.");

                csr_index_visited[csr_edge] = true;

                if (csr.column_indices[csr_edge] != target)
                    throw std::runtime_error("csc_to_csr target mismatch.");

                csc_matrix[source * neuron_count + target] +=
                    csr.weights[csr_edge];
            }
        }

        for (std::size_t i = 0; i < csr_matrix.size(); ++i)
        {
            if (std::fabs(csr_matrix[i] - csc_matrix[i]) > 1e-4f)
                throw std::runtime_error("CSR and CSC representations differ.");
        }

        std::cout << "CSR/CSC shared-weight consistency: PASS\n";

        // Select active neurons
        std::vector<snn::SpikeEvent> active_spikes;

        for (std::size_t neuron = 0; neuron < neuron_count; neuron += 10)
            active_spikes.push_back({neuron, 0});

        // Calculate expected currents using reference implementation
        std::vector<float> expected_currents(neuron_count, 0.0f);

        for (const auto& spike : active_spikes)
        {
            for (const auto& connection : connections)
            {
                if (connection.source == spike.neuron_id)
                    expected_currents[connection.target] += connection.weight;
            }
        }

        // Calculate currents using CSR event-driven processing
        std::vector<float> actual_currents(neuron_count, 0.0f);

        snn::processActiveSpikes(
            csr, active_spikes, actual_currents);

        // Compare results
        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            if (std::fabs(actual_currents[i] -
                          expected_currents[i]) > 1e-4f)
                throw std::runtime_error("Synaptic current mismatch.");
        }

        std::cout << "Event-driven current validation: PASS\n";
        std::cout << "Active spikes: " << active_spikes.size() << '\n';
        std::cout << "\nSTRESS TEST RESULT: PASS\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "STRESS TEST FAILED: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}