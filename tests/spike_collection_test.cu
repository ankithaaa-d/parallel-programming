#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace snn
{
    __global__ void atomicSpikeCollectionKernel(
        const int* spike_flags,
        int* active_spike_ids,
        int* active_spike_count,
        std::size_t neuron_count);
}

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
            throw std::runtime_error(cudaGetErrorString(error)); \
    } while (0)

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr
            << "Usage: spike_collection_test <spike_flag_1> "
            << "<spike_flag_2> ...\n";
        return 1;
    }

    int* d_flags = nullptr;
    int* d_active_ids = nullptr;
    int* d_active_count = nullptr;

    try
    {
        std::vector<int> spike_flags;

        for (int i = 1; i < argc; ++i)
        {
            std::string input = argv[i];

            if (input != "0" && input != "1")
                throw std::invalid_argument(
                    "Spike flags must be either 0 or 1.");

            spike_flags.push_back(std::stoi(input));
        }

        const std::size_t neuron_count = spike_flags.size();

        std::vector<int> active_ids(neuron_count);

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_flags),
            neuron_count * sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_active_ids),
            neuron_count * sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_active_count),
            sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            d_flags,
            spike_flags.data(),
            neuron_count * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(
            d_active_count, 0, sizeof(int)));

        const int threads_per_block = 256;

        const int blocks = static_cast<int>(
            (neuron_count + threads_per_block - 1)
            / threads_per_block);

        snn::atomicSpikeCollectionKernel<<<blocks, threads_per_block>>>(
            d_flags,
            d_active_ids,
            d_active_count,
            neuron_count);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int active_count = 0;

        CUDA_CHECK(cudaMemcpy(
            &active_count,
            d_active_count,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            active_ids.data(),
            d_active_ids,
            active_count * sizeof(int),
            cudaMemcpyDeviceToHost));

        // Validate count and IDs.
        std::vector<int> expected_ids;

        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            if (spike_flags[i] != 0)
                expected_ids.push_back(static_cast<int>(i));
        }

        std::sort(active_ids.begin(),
                  active_ids.begin() + active_count);

        const bool count_match =
            active_count == static_cast<int>(expected_ids.size());

        const bool ids_match =
            count_match &&
            std::equal(
                expected_ids.begin(),
                expected_ids.end(),
                active_ids.begin());

        std::cout << "Total neurons: " << neuron_count << '\n';
        std::cout << "Active spikes: " << active_count << '\n';

        std::cout << "Active neuron IDs: ";

        for (int i = 0; i < active_count; ++i)
            std::cout << active_ids[i] << " ";

        std::cout << "\n";

        std::cout << "Spike count validation: "
                  << (count_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Spike ID validation: "
                  << (ids_match ? "PASS" : "FAIL") << '\n';

        CUDA_CHECK(cudaFree(d_flags));
        CUDA_CHECK(cudaFree(d_active_ids));
        CUDA_CHECK(cudaFree(d_active_count));

        d_flags = nullptr;
        d_active_ids = nullptr;
        d_active_count = nullptr;

        return ids_match ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        if (d_flags) cudaFree(d_flags);
        if (d_active_ids) cudaFree(d_active_ids);
        if (d_active_count) cudaFree(d_active_count);

        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}