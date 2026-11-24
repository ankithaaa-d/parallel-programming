#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace snn
{
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
        int neuron_count);
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
            << "Usage: spike_compaction_test <flag1> <flag2> ...\n";
        return 1;
    }

    int* d_flags = nullptr;
    int* d_prefix = nullptr;
    int* d_compact_ids = nullptr;
    int* d_compact_count = nullptr;
    int* d_atomic_ids = nullptr;
    int* d_atomic_count = nullptr;

    try
    {
        std::vector<int> flags;

        for (int i = 1; i < argc; ++i)
        {
            const std::string input = argv[i];

            if (input != "0" && input != "1")
                throw std::invalid_argument(
                    "Spike flags must be 0 or 1.");

            flags.push_back(std::stoi(input));
        }

        const int neuron_count = static_cast<int>(flags.size());

        std::vector<int> prefix(neuron_count);
        std::vector<int> compact_ids(neuron_count);
        std::vector<int> atomic_ids(neuron_count);

        int compact_count = 0;
        int atomic_count = 0;

        const std::size_t bytes =
            flags.size() * sizeof(int);

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_flags), bytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_prefix), bytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_compact_ids), bytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_compact_count), sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_atomic_ids), bytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_atomic_count), sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            d_flags, flags.data(), bytes,
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_compact_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_atomic_count, 0, sizeof(int)));

        // Strategy 1: Inclusive prefix-sum compaction.
        CUDA_CHECK(snn::compactSpikesInclusive(
            d_flags,
            d_prefix,
            d_compact_ids,
            d_compact_count,
            neuron_count));

        // Strategy 2: Atomic append.
        const int threads_per_block = 256;

        const int blocks =
            (neuron_count + threads_per_block - 1)
            / threads_per_block;

        snn::atomicSpikeCollectionKernel<<<blocks, threads_per_block>>>(
            d_flags,
            d_atomic_ids,
            d_atomic_count,
            neuron_count);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy prefix-sum results.
        CUDA_CHECK(cudaMemcpy(
            prefix.data(), d_prefix, bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            compact_ids.data(), d_compact_ids, bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            &compact_count, d_compact_count, sizeof(int),
            cudaMemcpyDeviceToHost));

        // Copy atomic results.
        CUDA_CHECK(cudaMemcpy(
            &atomic_count, d_atomic_count, sizeof(int),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            atomic_ids.data(), d_atomic_ids,
            atomic_count * sizeof(int),
            cudaMemcpyDeviceToHost));

        // Build expected output from input flags.
        std::vector<int> expected_ids;

        for (int i = 0; i < neuron_count; ++i)
        {
            if (flags[i] == 1)
                expected_ids.push_back(i);
        }

        // Atomic output order is nondeterministic.
        std::sort(
            atomic_ids.begin(),
            atomic_ids.begin() + atomic_count);

        const bool compact_correct =
            compact_count == static_cast<int>(expected_ids.size()) &&
            std::equal(
                expected_ids.begin(),
                expected_ids.end(),
                compact_ids.begin());

        const bool atomic_correct =
            atomic_count == static_cast<int>(expected_ids.size()) &&
            std::equal(
                expected_ids.begin(),
                expected_ids.end(),
                atomic_ids.begin());

        std::cout << "\nTotal neurons: "
                  << neuron_count << '\n';

        std::cout << "\nInclusive prefix sum: ";

        for (int value : prefix)
            std::cout << value << " ";

        std::cout << "\n\nPrefix-sum active IDs: ";

        for (int i = 0; i < compact_count; ++i)
            std::cout << compact_ids[i] << " ";

        std::cout << "\nPrefix-sum spike count: "
                  << compact_count << '\n';

        std::cout << "\nAtomic active IDs (sorted): ";

        for (int i = 0; i < atomic_count; ++i)
            std::cout << atomic_ids[i] << " ";

        std::cout << "\nAtomic spike count: "
                  << atomic_count << '\n';

        std::cout << "\nPrefix-sum validation: "
                  << (compact_correct ? "PASS" : "FAIL") << '\n';

        std::cout << "Atomic validation: "
                  << (atomic_correct ? "PASS" : "FAIL") << '\n';

        CUDA_CHECK(cudaFree(d_flags));
        CUDA_CHECK(cudaFree(d_prefix));
        CUDA_CHECK(cudaFree(d_compact_ids));
        CUDA_CHECK(cudaFree(d_compact_count));
        CUDA_CHECK(cudaFree(d_atomic_ids));
        CUDA_CHECK(cudaFree(d_atomic_count));

        d_flags = nullptr;
        d_prefix = nullptr;
        d_compact_ids = nullptr;
        d_compact_count = nullptr;
        d_atomic_ids = nullptr;
        d_atomic_count = nullptr;

        return (compact_correct && atomic_correct) ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        if (d_flags) cudaFree(d_flags);
        if (d_prefix) cudaFree(d_prefix);
        if (d_compact_ids) cudaFree(d_compact_ids);
        if (d_compact_count) cudaFree(d_compact_count);
        if (d_atomic_ids) cudaFree(d_atomic_ids);
        if (d_atomic_count) cudaFree(d_atomic_count);

        std::cerr << "Test failed: "
                  << error.what() << '\n';

        return 1;
    }
}