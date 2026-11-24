#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>

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
        int neuron_count,
        int threads_per_block);
}

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
        {                                                     \
            throw std::runtime_error(cudaGetErrorString(error)); \
        }                                                     \
    } while (0)

struct TestData
{
    std::vector<int> spike_flags;
    std::vector<int> expected_prefix;
    std::vector<int> expected_ids;
};

TestData readInputFile(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Unable to open input file: " + filename);
    }

    TestData data;
    std::string token;

    int running_count = 0;

    while (file >> token)
    {
        std::size_t processed = 0;
        int flag;

        try
        {
            flag = std::stoi(token, &processed);
        }
        catch (...)
        {
            throw std::runtime_error(
                "Invalid spike flag in input file.");
        }

        if (processed != token.size() ||
            (flag != 0 && flag != 1))
        {
            throw std::runtime_error(
                "Spike flags must be either 0 or 1.");
        }

        if (data.spike_flags.size() >= INT_MAX)
        {
            throw std::runtime_error(
                "Input contains too many neurons.");
        }

        const int neuron_id =
            static_cast<int>(data.spike_flags.size());

        data.spike_flags.push_back(flag);

        running_count += flag;

        data.expected_prefix.push_back(running_count);

        if (flag == 1)
        {
            data.expected_ids.push_back(neuron_id);
        }
    }

    if (!file.eof())
    {
        throw std::runtime_error(
            "Error while reading input file.");
    }

    if (data.spike_flags.empty())
    {
        throw std::runtime_error(
            "Input file contains no neuron records.");
    }

    return data;
}

int readPositiveInteger(
    const char* input,
    const std::string& name)
{
    std::size_t processed = 0;

    int value;

    try
    {
        value = std::stoi(input, &processed);
    }
    catch (...)
    {
        throw std::invalid_argument(
            "Invalid value for " + name);
    }

    if (processed != std::string(input).size() ||
        value <= 0)
    {
        throw std::invalid_argument(
            name + " must be a positive integer.");
    }

    return value;
}

bool runStressTest(
    const TestData& data,
    int threads_per_block)
{
    int* d_flags = nullptr;
    int* d_prefix = nullptr;

    int* d_compact_ids = nullptr;
    int* d_compact_count = nullptr;

    int* d_atomic_ids = nullptr;
    int* d_atomic_count = nullptr;

    try
    {
        const int neuron_count =
            static_cast<int>(data.spike_flags.size());

        const std::size_t bytes =
            data.spike_flags.size() * sizeof(int);

        std::vector<int> prefix(neuron_count);
        std::vector<int> compact_ids(neuron_count);
        std::vector<int> atomic_ids(neuron_count);

        int compact_count = 0;
        int atomic_count = 0;

        // Allocate GPU memory.
        CUDA_CHECK(cudaMalloc(
            &d_flags, bytes));

        CUDA_CHECK(cudaMalloc(
            &d_prefix, bytes));

        CUDA_CHECK(cudaMalloc(
            &d_compact_ids, bytes));

        CUDA_CHECK(cudaMalloc(
            &d_compact_count, sizeof(int)));

        CUDA_CHECK(cudaMalloc(
            &d_atomic_ids, bytes));

        CUDA_CHECK(cudaMalloc(
            &d_atomic_count, sizeof(int)));

        // Copy input flags to GPU.
        CUDA_CHECK(cudaMemcpy(
            d_flags,
            data.spike_flags.data(),
            bytes,
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(
            d_compact_count, 0, sizeof(int)));

        CUDA_CHECK(cudaMemset(
            d_atomic_count, 0, sizeof(int)));

        // Strategy 1: Inclusive prefix-sum compaction.
        CUDA_CHECK(snn::compactSpikesInclusive(
            d_flags,
            d_prefix,
            d_compact_ids,
            d_compact_count,
            neuron_count,
            threads_per_block));

        // Strategy 2: Atomic append.
        const int blocks =
            (neuron_count + threads_per_block - 1)
            / threads_per_block;

        snn::atomicSpikeCollectionKernel
            <<<blocks, threads_per_block>>>(
                d_flags,
                d_atomic_ids,
                d_atomic_count,
                neuron_count);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy prefix-sum results.
        CUDA_CHECK(cudaMemcpy(
            prefix.data(),
            d_prefix,
            bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            &compact_count,
            d_compact_count,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        if (compact_count < 0 ||
            compact_count > neuron_count)
        {
            throw std::runtime_error(
                "Invalid prefix-sum spike count.");
        }

        if (compact_count > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                compact_ids.data(),
                d_compact_ids,
                compact_count * sizeof(int),
                cudaMemcpyDeviceToHost));
        }

        // Copy atomic results.
        CUDA_CHECK(cudaMemcpy(
            &atomic_count,
            d_atomic_count,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        if (atomic_count < 0 ||
            atomic_count > neuron_count)
        {
            throw std::runtime_error(
                "Invalid atomic spike count.");
        }

        if (atomic_count > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                atomic_ids.data(),
                d_atomic_ids,
                atomic_count * sizeof(int),
                cudaMemcpyDeviceToHost));
        }

        // Atomic append order is nondeterministic.
        std::sort(
            atomic_ids.begin(),
            atomic_ids.begin() + atomic_count);

        // Validate prefix sums.
        const bool prefix_match =
            prefix == data.expected_prefix;

        // Validate compacted neuron IDs.
        const bool compact_match =
            compact_count ==
                static_cast<int>(data.expected_ids.size()) &&
            std::equal(
                data.expected_ids.begin(),
                data.expected_ids.end(),
                compact_ids.begin());

        // Validate atomic neuron IDs.
        const bool atomic_match =
            atomic_count ==
                static_cast<int>(data.expected_ids.size()) &&
            std::equal(
                data.expected_ids.begin(),
                data.expected_ids.end(),
                atomic_ids.begin());

        const bool passed =
            prefix_match &&
            compact_match &&
            atomic_match;

        std::cout << "\n========== TEST RESULTS ==========\n";

        std::cout << "Neuron count: "
                  << neuron_count << '\n';

        std::cout << "CUDA block size: "
                  << threads_per_block << '\n';

        std::cout << "Expected spikes: "
                  << data.expected_ids.size() << '\n';

        std::cout << "Atomic spike count: "
                  << atomic_count << '\n';

        std::cout << "Prefix-sum spike count: "
                  << compact_count << '\n';

        std::cout << "Prefix-sum validation: "
                  << (prefix_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Compaction ID validation: "
                  << (compact_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Atomic ID validation: "
                  << (atomic_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Overall result: "
                  << (passed ? "PASS" : "FAIL") << '\n';

        // Release GPU memory.
        CUDA_CHECK(cudaFree(d_flags));
        CUDA_CHECK(cudaFree(d_prefix));
        CUDA_CHECK(cudaFree(d_compact_ids));
        CUDA_CHECK(cudaFree(d_compact_count));
        CUDA_CHECK(cudaFree(d_atomic_ids));
        CUDA_CHECK(cudaFree(d_atomic_count));

        return passed;
    }
    catch (...)
    {
        if (d_flags) cudaFree(d_flags);
        if (d_prefix) cudaFree(d_prefix);
        if (d_compact_ids) cudaFree(d_compact_ids);
        if (d_compact_count) cudaFree(d_compact_count);
        if (d_atomic_ids) cudaFree(d_atomic_ids);
        if (d_atomic_count) cudaFree(d_atomic_count);

        throw;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr
            << "Usage: spike_stress_test <input_file> "
            << "<threads_per_block>\n";

        return 1;
    }

    try
    {
        const std::string filename = argv[1];

        const int threads_per_block =
            readPositiveInteger(
                argv[2],
                "threads_per_block");

        // Verify requested block size against GPU capability.
        int device = 0;
        cudaDeviceProp properties{};

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(
            &properties, device));

        if (threads_per_block >
            properties.maxThreadsPerBlock)
        {
            throw std::invalid_argument(
                "Requested block size exceeds GPU limit.");
        }

        // Read runtime input.
        TestData data = readInputFile(filename);

        std::cout << "Input loaded successfully.\n";
        std::cout << "Number of neurons: "
                  << data.spike_flags.size() << '\n';

        const bool passed =
            runStressTest(data, threads_per_block);

        return passed ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Test failed: "
                  << error.what() << '\n';

        return 1;
    }
}