#include "connectivity.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

bool checkEqual(float a, float b)
{
    return std::fabs(a - b) < 1e-5f;
}

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void expectException(Function function, const char* message)
{
    bool thrown = false;

    try
    {
        function();
    }
    catch (const std::exception&)
    {
        thrown = true;
    }

    require(thrown, message);
}

int main()
{
    try
    {
        constexpr std::size_t neuron_count = 4;

        const std::vector<snn::SynapticConnection> connections =
        {
            {0, 1, 0.5f},
            {0, 2, 1.5f},
            {2, 1, 2.0f},
            {3, 0, 4.0f}
        };

        // 1. CSR construction and complete validation
        const auto csr = snn::buildCSR(neuron_count, connections);

        require(
            csr.row_offsets == std::vector<std::size_t>{0, 2, 2, 3, 4},
            "CSR offsets incorrect.");

        require(
            csr.column_indices == std::vector<snn::NeuronId>{1, 2, 1, 0},
            "CSR connection indices incorrect.");

        const std::vector<float> expected_csr_weights =
            {0.5f, 1.5f, 2.0f, 4.0f};

        for (std::size_t i = 0; i < expected_csr_weights.size(); ++i)
            require(checkEqual(csr.weights[i], expected_csr_weights[i]),
                    "CSR weights incorrect.");

        std::cout << "CSR construction and validation: PASS\n";

        // 2. CSC construction and shared-weight mapping validation
        const auto csc = snn::buildCSC(neuron_count, connections);

        require(
            csc.column_offsets == std::vector<std::size_t>{0, 1, 3, 4, 4},
            "CSC offsets incorrect.");

        require(
            csc.row_indices == std::vector<snn::NeuronId>{3, 0, 2, 0},
            "CSC connection indices incorrect.");

        require(
            csc.csc_to_csr == std::vector<std::size_t>{3, 0, 2, 1},
            "CSC-to-CSR index mapping incorrect.");

        const std::vector<float> expected_csc_weights =
            {4.0f, 0.5f, 2.0f, 1.5f};

        for (std::size_t i = 0; i < expected_csc_weights.size(); ++i)
        {
            const std::size_t csr_edge = csc.csc_to_csr[i];
            require(csr_edge < csr.weights.size(),
                    "CSC-to-CSR index out of bounds.");
            require(checkEqual(csr.weights[csr_edge], expected_csc_weights[i]),
                    "Shared CSC weight lookup incorrect.");
        }

        std::cout << "CSC construction and shared-weight validation: PASS\n";

        // 3. Event-driven active spike processing
        const std::vector<snn::SpikeEvent> active_spikes =
        {
            {0, 0},
            {2, 0}
        };

        std::vector<float> currents(neuron_count, 0.0f);

        snn::processActiveSpikes(csr, active_spikes, currents);

        const std::vector<float> expected_currents =
            {0.0f, 2.5f, 1.5f, 0.0f};

        for (std::size_t i = 0; i < neuron_count; ++i)
            require(checkEqual(currents[i], expected_currents[i]),
                    "Synaptic current mismatch.");

        std::cout << "Active spike processing: PASS\n";

        // 4. Empty connectivity
        const std::vector<snn::SynapticConnection> empty_connections;

        const auto empty_csr =
            snn::buildCSR(neuron_count, empty_connections);

        const auto empty_csc =
            snn::buildCSC(neuron_count, empty_connections);

        require(
            empty_csr.row_offsets ==
                std::vector<std::size_t>{0, 0, 0, 0, 0},
            "Empty CSR offsets incorrect.");

        require(
            empty_csc.column_offsets ==
                std::vector<std::size_t>{0, 0, 0, 0, 0},
            "Empty CSC offsets incorrect.");

        require(empty_csr.connection_count() == 0,
                "Empty CSR should contain no connections.");

        require(empty_csc.connection_count() == 0,
                "Empty CSC should contain no connections.");

        require(empty_csc.csc_to_csr.empty(),
                "Empty CSC mapping should be empty.");

        std::cout << "Empty connectivity: PASS\n";

        // 5. Invalid neuron IDs
        const std::vector<snn::SynapticConnection> invalid_connections =
        {
            {0, neuron_count, 1.0f}
        };

        expectException(
            [&]() { snn::buildCSR(neuron_count, invalid_connections); },
            "CSR accepted an invalid neuron ID.");

        expectException(
            [&]() { snn::buildCSC(neuron_count, invalid_connections); },
            "CSC accepted an invalid neuron ID.");

        std::cout << "Invalid neuron ID validation: PASS\n";

        // 6. Invalid synaptic weights
        const std::vector<snn::SynapticConnection> invalid_weights =
        {
            {0, 1, std::numeric_limits<float>::quiet_NaN()}
        };

        expectException(
            [&]() { snn::buildCSR(neuron_count, invalid_weights); },
            "CSR accepted a NaN weight.");

        expectException(
            [&]() { snn::buildCSC(neuron_count, invalid_weights); },
            "CSC accepted a NaN weight.");

        std::cout << "Invalid weight validation: PASS\n";

        // 7. Invalid active spike
        const std::vector<snn::SpikeEvent> invalid_spikes =
        {
            {neuron_count, 0}
        };

        std::vector<float> test_currents(neuron_count, 0.0f);

        expectException(
            [&]()
            {
                snn::processActiveSpikes(
                    csr, invalid_spikes, test_currents);
            },
            "Active spike processing accepted an invalid neuron ID.");

        std::cout << "Invalid active spike validation: PASS\n";

        std::cout << "\nOVERALL RESULT: PASS\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "\nTEST FAILED: " << error.what() << '\n';
        return 1;
    }

    return 0;
}