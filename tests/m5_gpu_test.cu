#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "connectivity.hpp"
#include "stdp_gpu.hpp"
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
                             "[%s] Mismatch at %zu: actual=%f expected=%f diff=%f\n",
                             label, i, actual[i], expected[i], diff);
                std::exit(EXIT_FAILURE);
            }
        }
        std::printf("  %-28s PASS (max_diff = %.2e)\n", label, max_diff);
    }

    void cpuReferenceSTDPStep(
        const snn::CSRConnectivity& csr,
        std::vector<float>& weights,
        std::vector<float>& x_pre,
        std::vector<float>& x_post,
        const std::vector<int>& active_spikes,
        float dt,
        const snn::STDPParams& params)
    {
        const float decay_pre = std::exp(-dt / params.tau_plus);
        const float decay_post = std::exp(-dt / params.tau_minus);

        for (std::size_t i = 0; i < x_pre.size(); ++i)
        {
            x_pre[i] *= decay_pre;
            x_post[i] *= decay_post;
        }

        std::vector<int> is_spiking(x_pre.size(), 0);
        for (int sid : active_spikes)
            is_spiking[static_cast<std::size_t>(sid)] = 1;

        for (std::size_t src = 0; src < csr.neuron_count(); ++src)
        {
            const std::size_t start = csr.row_offsets[src];
            const std::size_t end = csr.row_offsets[src + 1];
            for (std::size_t edge = start; edge < end; ++edge)
            {
                const std::size_t dst = csr.column_indices[edge];
                float dw = 0.0f;
                if (is_spiking[src])
                    dw -= params.a_minus * x_post[dst];
                if (is_spiking[dst])
                    dw += params.a_plus * x_pre[src];

                weights[edge] = std::min(params.w_max, std::max(params.w_min, weights[edge] + dw));
            }
        }

        for (int sid : active_spikes)
        {
            x_pre[static_cast<std::size_t>(sid)] += 1.0f;
            x_post[static_cast<std::size_t>(sid)] += 1.0f;
        }
    }

    void testCausalLTPAndLTD()
    {
        std::printf("\n=== Test 1: Causal LTP, LTD, Simultaneous Spikes & Bounds ===\n");

        const std::vector<snn::SynapticConnection> conns = {
            {0, 1, 0.50f},
            {1, 2, 0.005f}};

        snn::CSRConnectivity csr = snn::buildCSR(3, conns);
        snn::CSCConnectivity csc = snn::buildCSC(3, conns);

        snn::STDPParams params{};
        params.a_plus = 0.02f;
        params.a_minus = 0.03f;
        params.tau_plus = 20.0f;
        params.tau_minus = 20.0f;
        params.w_min = 0.0f;
        params.w_max = 1.0f;

        const float dt = 1.0f;
        const int threads = 128;

        std::vector<float> cpu_weights = csr.weights;
        std::vector<float> cpu_x_pre(3, 0.0f);
        std::vector<float> cpu_x_post(3, 0.0f);

        snn::DeviceCSR dev_csr{};
        snn::DeviceCSC dev_csc{};
        CUDA_CHECK(snn::uploadCSRToDevice(csr, dev_csr));
        CUDA_CHECK(snn::uploadCSCToDevice(csc, dev_csc));

        float* d_x_pre = nullptr;
        float* d_x_post = nullptr;
        int* d_spikes = nullptr;
        snn::SynapticEventRecord* d_events = nullptr;
        int* d_event_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_x_pre, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_post, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spikes, 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_events, 16 * sizeof(snn::SynapticEventRecord)));
        CUDA_CHECK(cudaMalloc(&d_event_count, sizeof(int)));

        CUDA_CHECK(cudaMemset(d_x_pre, 0, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_x_post, 0, 3 * sizeof(float)));

        const std::vector<std::vector<int>> schedule = {
            {0},
            {1},
            {0},
            {1, 2},
            {1}};

        for (std::size_t step = 0; step < schedule.size(); ++step)
        {
            const std::vector<int>& spikes = schedule[step];
            const int spike_count = static_cast<int>(spikes.size());

            cpuReferenceSTDPStep(csr, cpu_weights, cpu_x_pre, cpu_x_post, spikes, dt, params);

            CUDA_CHECK(cudaMemcpy(d_spikes, spikes.data(),
                                  spike_count * sizeof(int), cudaMemcpyHostToDevice));

            CUDA_CHECK(snn::decaySTDPTracesGPU(
                d_x_pre, d_x_post, 3, dt, params.tau_plus, params.tau_minus, threads));

            CUDA_CHECK(snn::processActiveSpikesGPU(
                dev_csr, dev_csc, d_spikes, spike_count,
                d_x_pre, d_x_post, params.a_plus, params.a_minus,
                d_events, d_event_count, 16, threads));

            int host_event_count = 0;
            CUDA_CHECK(cudaMemcpy(&host_event_count, d_event_count, sizeof(int), cudaMemcpyDeviceToHost));

            CUDA_CHECK(snn::applySTDPWeightUpdatesGPU(
                d_events, host_event_count, dev_csr.d_weights, dev_csr.connection_count,
                params.w_min, params.w_max, threads));

            CUDA_CHECK(snn::incrementSTDPTracesGPU(
                d_x_pre, d_x_post, d_spikes, spike_count, 1.0f, 1.0f, threads));
        }

        std::vector<float> gpu_weights(2, 0.0f);
        std::vector<float> gpu_x_pre(3, 0.0f);
        std::vector<float> gpu_x_post(3, 0.0f);

        CUDA_CHECK(cudaMemcpy(gpu_weights.data(), dev_csr.d_weights, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_x_pre.data(), d_x_pre, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_x_post.data(), d_x_post, 3 * sizeof(float), cudaMemcpyDeviceToHost));

        verifyClose(gpu_weights, cpu_weights, 1e-6f, "Causal Weights (LTP/LTD)");
        verifyClose(gpu_x_pre, cpu_x_pre, 1e-6f, "Presynaptic Traces");
        verifyClose(gpu_x_post, cpu_x_post, 1e-6f, "Postsynaptic Traces");
        assert(gpu_weights[0] > 0.48f && gpu_weights[0] < 0.65f);
        assert(std::fabs(gpu_weights[1] - 0.0f) < 1e-6f);

        CUDA_CHECK(cudaFree(d_x_pre));
        CUDA_CHECK(cudaFree(d_x_post));
        CUDA_CHECK(cudaFree(d_spikes));
        CUDA_CHECK(cudaFree(d_events));
        CUDA_CHECK(cudaFree(d_event_count));
        snn::freeDeviceCSR(dev_csr);
        snn::freeDeviceCSC(dev_csc);
    }

    void testMultiStepNetworkParity()
    {
        std::printf("\n=== Test 2: 25-Step Random Network STDP Parity (N=512, p=0.08) ===\n");

        const int neuron_count = 512;
        const float connection_prob = 0.08f;
        const float spike_prob = 0.12f;
        const int steps = 25;
        const float dt = 1.0f;
        const int threads = 256;

        std::srand(314);
        std::vector<snn::SynapticConnection> conns;
        for (int src = 0; src < neuron_count; ++src)
        {
            for (int dst = 0; dst < neuron_count; ++dst)
            {
                if (src == dst)
                    continue;
                if ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) < connection_prob)
                {
                    const float w = 0.1f + 0.4f * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
                    conns.push_back({static_cast<snn::NeuronId>(src),
                                     static_cast<snn::NeuronId>(dst),
                                     w});
                }
            }
        }

        snn::CSRConnectivity csr = snn::buildCSR(neuron_count, conns);
        snn::CSCConnectivity csc = snn::buildCSC(neuron_count, conns);

        snn::STDPParams params{};
        std::vector<float> cpu_weights = csr.weights;
        std::vector<float> cpu_x_pre(neuron_count, 0.0f);
        std::vector<float> cpu_x_post(neuron_count, 0.0f);

        snn::DeviceCSR dev_csr{};
        snn::DeviceCSC dev_csc{};
        CUDA_CHECK(snn::uploadCSRToDevice(csr, dev_csr));
        CUDA_CHECK(snn::uploadCSCToDevice(csc, dev_csc));

        const int max_events = std::max(1, dev_csr.connection_count * 2);

        float* d_x_pre = nullptr;
        float* d_x_post = nullptr;
        int* d_spikes = nullptr;
        snn::SynapticEventRecord* d_events = nullptr;
        int* d_event_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_x_pre, neuron_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_post, neuron_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spikes, neuron_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_events, max_events * sizeof(snn::SynapticEventRecord)));
        CUDA_CHECK(cudaMalloc(&d_event_count, sizeof(int)));

        CUDA_CHECK(cudaMemset(d_x_pre, 0, neuron_count * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_x_post, 0, neuron_count * sizeof(float)));

        for (int t = 0; t < steps; ++t)
        {
            std::vector<int> active_spikes;
            for (int i = 0; i < neuron_count; ++i)
            {
                if ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) < spike_prob)
                    active_spikes.push_back(i);
            }
            const int spike_count = static_cast<int>(active_spikes.size());

            cpuReferenceSTDPStep(csr, cpu_weights, cpu_x_pre, cpu_x_post, active_spikes, dt, params);

            if (spike_count > 0)
            {
                CUDA_CHECK(cudaMemcpy(d_spikes, active_spikes.data(),
                                      spike_count * sizeof(int), cudaMemcpyHostToDevice));
            }

            CUDA_CHECK(snn::decaySTDPTracesGPU(
                d_x_pre, d_x_post, neuron_count, dt, params.tau_plus, params.tau_minus, threads));

            CUDA_CHECK(snn::processActiveSpikesGPU(
                dev_csr, dev_csc, d_spikes, spike_count,
                d_x_pre, d_x_post, params.a_plus, params.a_minus,
                d_events, d_event_count, max_events, threads));

            int host_event_count = 0;
            CUDA_CHECK(cudaMemcpy(&host_event_count, d_event_count, sizeof(int), cudaMemcpyDeviceToHost));
            assert(host_event_count <= max_events);

            CUDA_CHECK(snn::applySTDPWeightUpdatesGPU(
                d_events, host_event_count, dev_csr.d_weights, dev_csr.connection_count,
                params.w_min, params.w_max, threads));

            CUDA_CHECK(snn::incrementSTDPTracesGPU(
                d_x_pre, d_x_post, d_spikes, spike_count, 1.0f, 1.0f, threads));
        }

        std::vector<float> gpu_weights(dev_csr.connection_count, 0.0f);
        std::vector<float> gpu_x_pre(neuron_count, 0.0f);
        std::vector<float> gpu_x_post(neuron_count, 0.0f);

        CUDA_CHECK(cudaMemcpy(gpu_weights.data(), dev_csr.d_weights,
                              dev_csr.connection_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_x_pre.data(), d_x_pre,
                              neuron_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_x_post.data(), d_x_post,
                              neuron_count * sizeof(float), cudaMemcpyDeviceToHost));

        verifyClose(gpu_weights, cpu_weights, 1e-5f, "25-Step CSR Weights");
        verifyClose(gpu_x_pre, cpu_x_pre, 1e-6f, "25-Step x_pre Traces");
        verifyClose(gpu_x_post, cpu_x_post, 1e-6f, "25-Step x_post Traces");

        CUDA_CHECK(cudaFree(d_x_pre));
        CUDA_CHECK(cudaFree(d_x_post));
        CUDA_CHECK(cudaFree(d_spikes));
        CUDA_CHECK(cudaFree(d_events));
        CUDA_CHECK(cudaFree(d_event_count));
        snn::freeDeviceCSR(dev_csr);
        snn::freeDeviceCSC(dev_csc);
    }
} // namespace

int main()
{
    std::printf("=== M5 GPU STDP Trace Dynamics & Weight Updates Validation ===\n");
    testCausalLTPAndLTD();
    testMultiStepNetworkParity();
    std::printf("\nAll M5 GPU STDP tests validated successfully!\n");
    return 0;
}