#include <cuda_runtime.h>

#include "common_types.hpp"
#include "neuron_model.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
            throw std::runtime_error(cudaGetErrorString(error)); \
    } while (0)

namespace snn
{
    __global__ void neuronUpdateKernel(
        NeuronState* states,
        const NeuronParameters* parameters,
        const float* input_currents,
        int* spike_flags,
        std::size_t neuron_count,
        float time_step);

    __global__ void atomicSpikeCollectionKernel(
        const int* spike_flags,
        int* active_spike_ids,
        int* active_spike_count,
        std::size_t neuron_count);

    cudaError_t compactSpikesInclusive(
        const int* d_spike_flags,
        int* d_prefix_sum,
        int* d_active_spike_ids,
        int* d_active_spike_count,
        int neuron_count,
        int threads_per_block);
}

struct NeuronInput
{
    snn::NeuronParameters parameters;
    snn::NeuronState state;
    float current;
};

float readFiniteFloat(std::ifstream& file, const std::string& name)
{
    float value;

    if (!(file >> value) || !std::isfinite(value))
        throw std::runtime_error("Invalid or missing input: " + name);

    return value;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr
            << "Usage: m2_integration_test <input_file> "
            << "<threads_per_block>\n";

        return 1;
    }

    try
    {
        const std::string input_path = argv[1];

        std::size_t processed = 0;
        int threads_per_block = std::stoi(argv[2], &processed);

        if (processed != std::string(argv[2]).size() ||
            threads_per_block <= 0)
        {
            throw std::invalid_argument("Invalid CUDA block size.");
        }

        int device = 0;
        cudaDeviceProp device_properties{};

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(
            &device_properties, device));

        if (threads_per_block > device_properties.maxThreadsPerBlock)
            throw std::invalid_argument(
                "Block size exceeds device maximum.");

        // ------------------------------------------------
        // 1. Read runtime input
        // ------------------------------------------------

        std::ifstream input_file(input_path);

        if (!input_file)
            throw std::runtime_error("Cannot open input file.");

        std::size_t neuron_count;
        float time_step;

        if (!(input_file >> neuron_count >> time_step))
            throw std::runtime_error("Invalid input file header.");

        if (neuron_count == 0 ||
            neuron_count > static_cast<std::size_t>(INT_MAX))
        {
            throw std::runtime_error("Invalid neuron count.");
        }

        if (!std::isfinite(time_step) || time_step <= 0.0f)
            throw std::runtime_error("Invalid time step.");

        std::vector<NeuronInput> inputs(neuron_count);

        std::vector<snn::NeuronState> cpu_states(neuron_count);
        std::vector<snn::NeuronState> gpu_states(neuron_count);

        std::vector<snn::NeuronParameters> parameters(neuron_count);
        std::vector<float> currents(neuron_count);

        std::vector<int> cpu_spike_flags(neuron_count, 0);
        std::vector<int> gpu_spike_flags(neuron_count, 0);

        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            auto& item = inputs[i];

            item.parameters.a = readFiniteFloat(input_file, "a");
            item.parameters.b = readFiniteFloat(input_file, "b");
            item.parameters.c = readFiniteFloat(input_file, "c");
            item.parameters.d = readFiniteFloat(input_file, "d");

            item.state.voltage =
                readFiniteFloat(input_file, "voltage");

            item.state.recovery =
                readFiniteFloat(input_file, "recovery");

            item.current =
                readFiniteFloat(input_file, "current");

            parameters[i] = item.parameters;
            currents[i] = item.current;

            cpu_states[i] = item.state;
            gpu_states[i] = item.state;
        }

        std::string extra_data;

        if (input_file >> extra_data)
            throw std::runtime_error(
                "Unexpected extra data in input file.");

        std::cout << "Input loaded successfully.\n";
        std::cout << "Neuron count: " << neuron_count << '\n';
        std::cout << "CUDA block size: "
                  << threads_per_block << '\n';

        // ------------------------------------------------
        // 2. CPU reference execution
        // ------------------------------------------------

        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            const bool spike = snn::updateNeuron(
                cpu_states[i],
                parameters[i],
                currents[i],
                time_step);

            cpu_spike_flags[i] = spike ? 1 : 0;
        }

        // ------------------------------------------------
        // 3. Allocate GPU memory
        // ------------------------------------------------

        snn::NeuronState* d_states = nullptr;
        snn::NeuronParameters* d_parameters = nullptr;

        float* d_currents = nullptr;
        int* d_spike_flags = nullptr;

        int* d_atomic_ids = nullptr;
        int* d_atomic_count = nullptr;

        int* d_prefix_sum = nullptr;
        int* d_compact_ids = nullptr;
        int* d_compact_count = nullptr;

        try
        {
            const std::size_t state_bytes =
                neuron_count * sizeof(snn::NeuronState);

            const std::size_t parameter_bytes =
                neuron_count * sizeof(snn::NeuronParameters);

            const std::size_t float_bytes =
                neuron_count * sizeof(float);

            const std::size_t int_bytes =
                neuron_count * sizeof(int);

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_states),
                state_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_parameters),
                parameter_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_currents),
                float_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_spike_flags),
                int_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_atomic_ids),
                int_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_atomic_count),
                sizeof(int)));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_prefix_sum),
                int_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_compact_ids),
                int_bytes));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_compact_count),
                sizeof(int)));

            // Copy identical initial inputs to GPU.

            CUDA_CHECK(cudaMemcpy(
                d_states,
                gpu_states.data(),
                state_bytes,
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_parameters,
                parameters.data(),
                parameter_bytes,
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_currents,
                currents.data(),
                float_bytes,
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemset(
                d_spike_flags, 0, int_bytes));

            CUDA_CHECK(cudaMemset(
                d_atomic_count, 0, sizeof(int)));

            CUDA_CHECK(cudaMemset(
                d_compact_count, 0, sizeof(int)));

            // ------------------------------------------------
            // 4. GPU neuron update
            // ------------------------------------------------

            const int blocks =
                static_cast<int>(
                    (neuron_count + threads_per_block - 1)
                    / threads_per_block);

            snn::neuronUpdateKernel<<<blocks, threads_per_block>>>(
                d_states,
                d_parameters,
                d_currents,
                d_spike_flags,
                neuron_count,
                time_step);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(
                gpu_states.data(),
                d_states,
                state_bytes,
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                gpu_spike_flags.data(),
                d_spike_flags,
                int_bytes,
                cudaMemcpyDeviceToHost));

            // ------------------------------------------------
            // 5. Compare CPU and GPU neuron results
            // ------------------------------------------------

            const float tolerance = 1e-4f;

            bool state_match = true;
            bool spike_match = true;

            for (std::size_t i = 0; i < neuron_count; ++i)
            {
                if (std::fabs(
                    cpu_states[i].voltage -
                    gpu_states[i].voltage) > tolerance ||
                    std::fabs(
                    cpu_states[i].recovery -
                    gpu_states[i].recovery) > tolerance)
                {
                    state_match = false;
                }

                if (cpu_spike_flags[i] != gpu_spike_flags[i])
                    spike_match = false;
            }

            // ------------------------------------------------
            // 6. Atomic spike collection
            // ------------------------------------------------

            snn::atomicSpikeCollectionKernel
                <<<blocks, threads_per_block>>>(
                    d_spike_flags,
                    d_atomic_ids,
                    d_atomic_count,
                    neuron_count);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int atomic_count = 0;

            std::vector<int> atomic_ids(neuron_count);

            CUDA_CHECK(cudaMemcpy(
                &atomic_count,
                d_atomic_count,
                sizeof(int),
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                atomic_ids.data(),
                d_atomic_ids,
                int_bytes,
                cudaMemcpyDeviceToHost));

            atomic_ids.resize(atomic_count);
            std::sort(atomic_ids.begin(), atomic_ids.end());

            // ------------------------------------------------
            // 7. Prefix-sum spike compaction
            // ------------------------------------------------

            CUDA_CHECK(snn::compactSpikesInclusive(
                d_spike_flags,
                d_prefix_sum,
                d_compact_ids,
                d_compact_count,
                static_cast<int>(neuron_count),
                threads_per_block));

            CUDA_CHECK(cudaDeviceSynchronize());

            int compact_count = 0;

            std::vector<int> compact_ids(neuron_count);

            CUDA_CHECK(cudaMemcpy(
                &compact_count,
                d_compact_count,
                sizeof(int),
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                compact_ids.data(),
                d_compact_ids,
                int_bytes,
                cudaMemcpyDeviceToHost));

            compact_ids.resize(compact_count);

            // ------------------------------------------------
            // 8. Independently calculate expected spike IDs
            // ------------------------------------------------

            std::vector<int> expected_ids;

            for (std::size_t i = 0; i < neuron_count; ++i)
            {
                if (cpu_spike_flags[i] != 0)
                    expected_ids.push_back(static_cast<int>(i));
            }

            const bool atomic_match =
                atomic_ids == expected_ids;

            const bool compact_match =
                compact_ids == expected_ids;

            // ------------------------------------------------
            // 9. Results
            // ------------------------------------------------

            std::cout << "\n========== M2 INTEGRATION RESULTS ==========\n";

            std::cout << "Neuron count: "
                      << neuron_count << '\n';

            std::cout << "Expected active spikes: "
                      << expected_ids.size() << '\n';

            std::cout << "Atomic spike count: "
                      << atomic_count << '\n';

            std::cout << "Prefix-sum spike count: "
                      << compact_count << '\n';

            std::cout << "\nCPU-GPU voltage/recovery comparison: "
                      << (state_match ? "PASS" : "FAIL") << '\n';

            std::cout << "CPU-GPU spike flag comparison: "
                      << (spike_match ? "PASS" : "FAIL") << '\n';

            std::cout << "Atomic collection validation: "
                      << (atomic_match ? "PASS" : "FAIL") << '\n';

            std::cout << "Prefix-sum compaction validation: "
                      << (compact_match ? "PASS" : "FAIL") << '\n';

            const bool overall =
                state_match &&
                spike_match &&
                atomic_match &&
                compact_match;

            std::cout << "\nOverall result: "
                      << (overall ? "PASS" : "FAIL") << '\n';

            // ------------------------------------------------
            // 10. Free GPU memory
            // ------------------------------------------------

            CUDA_CHECK(cudaFree(d_states));
            d_states = nullptr;

            CUDA_CHECK(cudaFree(d_parameters));
            d_parameters = nullptr;

            CUDA_CHECK(cudaFree(d_currents));
            d_currents = nullptr;

            CUDA_CHECK(cudaFree(d_spike_flags));
            d_spike_flags = nullptr;

            CUDA_CHECK(cudaFree(d_atomic_ids));
            d_atomic_ids = nullptr;

            CUDA_CHECK(cudaFree(d_atomic_count));
            d_atomic_count = nullptr;

            CUDA_CHECK(cudaFree(d_prefix_sum));
            d_prefix_sum = nullptr;

            CUDA_CHECK(cudaFree(d_compact_ids));
            d_compact_ids = nullptr;

            CUDA_CHECK(cudaFree(d_compact_count));
            d_compact_count = nullptr;

            if (!overall)
                return 1;
        }
        catch (...)
        {
            if (d_states) cudaFree(d_states);
            if (d_parameters) cudaFree(d_parameters);
            if (d_currents) cudaFree(d_currents);
            if (d_spike_flags) cudaFree(d_spike_flags);
            if (d_atomic_ids) cudaFree(d_atomic_ids);
            if (d_atomic_count) cudaFree(d_atomic_count);
            if (d_prefix_sum) cudaFree(d_prefix_sum);
            if (d_compact_ids) cudaFree(d_compact_ids);
            if (d_compact_count) cudaFree(d_compact_count);

            throw;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "\nError: " << error.what() << '\n';
        return 1;
    }

    return 0;
}