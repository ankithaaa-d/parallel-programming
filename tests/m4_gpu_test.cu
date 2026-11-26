#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "connectivity.hpp"
#include "current_accumulation.hpp"
#include "synaptic_events.hpp"

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err__));                \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace
{
    std::vector<snn::SynapticConnection> generateConnections(
        int neuron_count,
        float connection_prob,
        float min_weight,
        float max_weight,
        unsigned int seed)
    {
        std::srand(seed);
        std::vector<snn::SynapticConnection> connections;
        for (int src = 0; src < neuron_count; ++src)
        {
            for (int dst = 0; dst < neuron_count; ++dst)
            {
                if (src == dst)
                    continue;
                const float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                if (r < connection_prob)
                {
                    const float w_rand = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                    const float w = min_weight + w_rand * (max_weight - min_weight);
                    connections.push_back({static_cast<snn::NeuronId>(src),
                                           static_cast<snn::NeuronId>(dst),
                                           w});
                }
            }
        }
        return connections;
    }

    std::vector<float> computeReferenceCurrents(
        const snn::CSRConnectivity& csr,
        const std::vector<int>& spike_flags)
    {
        const std::size_t n = csr.neuron_count();
        std::vector<float> ref(n, 0.0f);
        for (std::size_t src = 0; src < n; ++src)
        {
            if (spike_flags[src] == 0)
                continue;

            const std::size_t start = csr.row_offsets[src];
            const std::size_t end = csr.row_offsets[src + 1];
            for (std::size_t edge = start; edge < end; ++edge)
            {
                const snn::NeuronId dst = csr.column_indices[edge];
                ref[dst] += csr.weights[edge];
            }
        }
        return ref;
    }

    void verifyClose(
        const std::vector<float>& actual,
        const std::vector<float>& expected,
        float tol,
        const char* label)
    {
        assert(actual.size() == expected.size());
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            const float diff = std::fabs(actual[i] - expected[i]);
            if (diff > max_diff)
                max_diff = diff;
            if (diff > tol)
            {
                std::fprintf(stderr,
                             "[%s] Mismatch at neuron %zu: actual=%f expected=%f diff=%f\n",
                             label, i, actual[i], expected[i], diff);
                std::exit(EXIT_FAILURE);
            }
        }
        std::printf("  %-22s PASS (max_diff = %.2e)\n", label, max_diff);
    }

    void runBenchmarkAndValidation(
        const char* regime_name,
        int neuron_count,
        float connection_prob,
        float spike_prob,
        unsigned int seed)
    {
        std::printf("\n=== Regime: %s (N=%d, p=%.3f, spike_prob=%.2f) ===\n",
                    regime_name, neuron_count, connection_prob, spike_prob);

        const std::vector<snn::SynapticConnection> connections =
            generateConnections(neuron_count, connection_prob, 0.05f, 0.25f, seed);

        snn::CSRConnectivity csr = snn::buildCSR(static_cast<std::size_t>(neuron_count), connections);
        snn::CSCConnectivity csc = snn::buildCSC(static_cast<std::size_t>(neuron_count), connections);

        std::srand(seed + 99);
        std::vector<int> host_spike_flags(neuron_count, 0);
        std::vector<int> active_spike_ids;
        for (int i = 0; i < neuron_count; ++i)
        {
            const float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            if (r < spike_prob)
            {
                host_spike_flags[i] = 1;
                active_spike_ids.push_back(i);
            }
        }
        const int active_spike_count = static_cast<int>(active_spike_ids.size());

        const std::vector<float> expected = computeReferenceCurrents(csr, host_spike_flags);

        snn::DeviceCSR dev_csr{};
        snn::DeviceCSC dev_csc{};
        CUDA_CHECK(snn::uploadCSRToDevice(csr, dev_csr));
        CUDA_CHECK(snn::uploadCSCToDevice(csc, dev_csc));

        int* d_spike_flags = nullptr;
        int* d_active_spike_ids = nullptr;
        int* d_target_hit_counts = nullptr;
        float* d_synaptic_currents = nullptr;
        snn::SynapticEventRecord* d_events = nullptr;
        int* d_event_count = nullptr;

        const int conn_count = static_cast<int>(csr.connection_count());
        const int max_events = std::max(1, conn_count * 2);

        CUDA_CHECK(cudaMalloc(&d_spike_flags, neuron_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_active_spike_ids, std::max(1, active_spike_count) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_target_hit_counts, neuron_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_synaptic_currents, neuron_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_events, max_events * sizeof(snn::SynapticEventRecord)));
        CUDA_CHECK(cudaMalloc(&d_event_count, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_spike_flags, host_spike_flags.data(),
                              neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        if (active_spike_count > 0)
        {
            CUDA_CHECK(cudaMemcpy(d_active_spike_ids, active_spike_ids.data(),
                                  active_spike_count * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(d_event_count, 0, sizeof(int)));

        const int threads = 256;
        CUDA_CHECK(snn::processActiveSpikesGPU(
            dev_csr,
            dev_csc,
            d_active_spike_ids,
            active_spike_count,
            nullptr,
            nullptr,
            0.01f,
            0.012f,
            d_events,
            d_event_count,
            max_events,
            threads));
        CUDA_CHECK(cudaDeviceSynchronize());

        int host_event_count = 0;
        CUDA_CHECK(cudaMemcpy(&host_event_count, d_event_count, sizeof(int), cudaMemcpyDeviceToHost));
        assert(host_event_count <= max_events);

        snn::CurrentContentionMetrics metrics{};
        CUDA_CHECK(snn::measureCurrentContentionGPU(
            d_events, host_event_count, neuron_count, d_target_hit_counts, metrics, threads));
        std::printf("  Active spikes: %d | Current events: %d | Unique targets: %d | Max hits: %d | C_I: %.2f\n",
                    active_spike_count,
                    metrics.total_current_events,
                    metrics.unique_targets_updated,
                    metrics.max_updates_per_target,
                    metrics.contention_ratio_ci);

        std::vector<float> actual(neuron_count, 0.0f);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const snn::CurrentAccumulationStrategy strategies[3] = {
            snn::CurrentAccumulationStrategy::Atomic,
            snn::CurrentAccumulationStrategy::BlockAggregated,
            snn::CurrentAccumulationStrategy::Gather};
        const char* names[3] = {
            "Strategy 1 (Atomic)",
            "Strategy 2 (Aggregated)",
            "Strategy 3 (Gather)"};

        for (int s = 0; s < 3; ++s)
        {
            // Warmup
            CUDA_CHECK(snn::accumulatePostsynapticCurrentsGPU(
                strategies[s], dev_csr, dev_csc, d_spike_flags,
                d_events, host_event_count, d_synaptic_currents, threads));
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start));
            for (int iter = 0; iter < 50; ++iter)
            {
                CUDA_CHECK(snn::accumulatePostsynapticCurrentsGPU(
                    strategies[s], dev_csr, dev_csc, d_spike_flags,
                    d_events, host_event_count, d_synaptic_currents, threads));
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

            CUDA_CHECK(cudaMemcpy(actual.data(), d_synaptic_currents,
                                  neuron_count * sizeof(float), cudaMemcpyDeviceToHost));
            verifyClose(actual, expected, 1e-4f, names[s]);
            std::printf("    Avg kernel time (50 runs): %.4f ms\n", ms / 50.0f);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_spike_flags));
        CUDA_CHECK(cudaFree(d_active_spike_ids));
        CUDA_CHECK(cudaFree(d_target_hit_counts));
        CUDA_CHECK(cudaFree(d_synaptic_currents));
        CUDA_CHECK(cudaFree(d_events));
        CUDA_CHECK(cudaFree(d_event_count));
        snn::freeDeviceCSR(dev_csr);
        snn::freeDeviceCSC(dev_csc);
    }
} // namespace

int main()
{
    std::printf("=== M4 GPU Postsynaptic Current Accumulation Validation ===\n");

    // 1. Low contention regime (shared-memory path: N <= 4096, sparse spikes)
    runBenchmarkAndValidation("Low Contention (Sparse, Shared-Mem Path)", 1024, 0.05f, 0.03f, 42);

    // 2. High contention regime (shared-memory path: N <= 4096, dense burst)
    runBenchmarkAndValidation("High Contention (Burst, Shared-Mem Path)", 1024, 0.25f, 0.60f, 108);

    // 3. Large network regime (warp-aggregated path: N > 4096)
    runBenchmarkAndValidation("Large Network (Warp-Aggregated Path)", 8192, 0.04f, 0.25f, 777);

    std::printf("\nAll M4 GPU current accumulation strategies validated successfully!\n");
    return 0;
}