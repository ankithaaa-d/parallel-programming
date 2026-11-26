#include <cuda_runtime.h>

#include "common_types.hpp"
#include "connectivity.hpp"
#include "neuron_model.hpp"
#include "synaptic_events.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t error = (call);                                \
        if (error != cudaSuccess)                                  \
            throw std::runtime_error(cudaGetErrorString(error));   \
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

    cudaError_t compactSpikesInclusive(
        const int* d_spike_flags,
        int* d_prefix_sum,
        int* d_active_spike_ids,
        int* d_active_spike_count,
        int neuron_count,
        int threads_per_block);
}

bool eventLess(const snn::SynapticEventRecord& a, const snn::SynapticEventRecord& b)
{
    return std::tie(a.event_type, a.synapse_id, a.source, a.target) <
           std::tie(b.event_type, b.synapse_id, b.source, b.target);
}

int main()
{
    try
    {
        constexpr std::size_t neuron_count = 1024;
        constexpr std::size_t connection_count = 10000;
        constexpr float time_step = 1.0f;
        constexpr int threads_per_block = 256;
        constexpr float a_plus = 0.01f;
        constexpr float a_minus = 0.012f;
        constexpr unsigned int seed = 12345;

        std::mt19937 gen(seed);
        std::uniform_int_distribution<std::size_t> neuron_dist(0, neuron_count - 1);
        std::uniform_real_distribution<float> weight_dist(0.1f, 3.0f);
        std::uniform_real_distribution<float> current_dist(0.0f, 200.0f);
        std::uniform_real_distribution<float> trace_dist(0.0f, 1.0f);

        // 1. Build network & traces
        std::vector<snn::SynapticConnection> connections;
        connections.reserve(connection_count);
        for (std::size_t i = 0; i < connection_count; ++i)
        {
            connections.push_back({
                neuron_dist(gen),
                neuron_dist(gen),
                weight_dist(gen)
            });
        }

        const auto csr = snn::buildCSR(neuron_count, connections);
        const auto csc = snn::buildCSC(neuron_count, connections);

        std::vector<snn::NeuronParameters> params(neuron_count, {0.02f, 0.2f, -65.0f, 8.0f});
        std::vector<snn::NeuronState> cpu_states(neuron_count, {-65.0f, -13.0f});
        std::vector<float> input_currents(neuron_count);
        std::vector<float> x_pre(neuron_count);
        std::vector<float> x_post(neuron_count);

        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            input_currents[i] = current_dist(gen);
            x_pre[i] = trace_dist(gen);
            x_post[i] = trace_dist(gen);
        }

        // 2. CPU Reference: M2 neuron update + M3 CSR/CSC event generation
        std::vector<snn::SpikeEvent> cpu_spikes;
        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            if (snn::updateNeuron(cpu_states[i], params[i], input_currents[i], time_step))
                cpu_spikes.push_back({i, 0});
        }

        std::vector<snn::SynapticEventRecord> expected_events;
        for (const auto& spike : cpu_spikes)
        {
            const std::size_t src = spike.neuron_id;
            for (std::size_t edge = csr.row_offsets[src]; edge < csr.row_offsets[src + 1]; ++edge)
            {
                const std::size_t tgt = csr.column_indices[edge];
                expected_events.push_back(snn::SynapticEventRecord{
                    static_cast<int>(src),
                    static_cast<int>(tgt),
                    static_cast<int>(edge),
                    csr.weights[edge],
                    -a_minus * x_post[tgt],
                    static_cast<int>(snn::SynapticEventType::PresynapticCSR)
                });
            }
        }

        for (const auto& spike : cpu_spikes)
        {
            const std::size_t tgt = spike.neuron_id;
            for (std::size_t edge = csc.column_offsets[tgt]; edge < csc.column_offsets[tgt + 1]; ++edge)
            {
                const std::size_t src = csc.row_indices[edge];
                const std::size_t csr_id = csc.csc_to_csr[edge];
                expected_events.push_back(snn::SynapticEventRecord{
                    static_cast<int>(src),
                    static_cast<int>(tgt),
                    static_cast<int>(csr_id),
                    0.0f,
                    a_plus * x_pre[src],
                    static_cast<int>(snn::SynapticEventType::PostsynapticCSC)
                });
            }
        }

        std::vector<float> expected_currents(neuron_count, 0.0f);
        snn::processActiveSpikes(csr, cpu_spikes, expected_currents);

        // 3. GPU Execution: M2 -> M3 pipeline
        snn::DeviceCSR dev_csr{};
        snn::DeviceCSC dev_csc{};
        CUDA_CHECK(snn::uploadCSRToDevice(csr, dev_csr));
        CUDA_CHECK(snn::uploadCSCToDevice(csc, dev_csc));

        snn::NeuronState* d_states = nullptr;
        snn::NeuronParameters* d_params = nullptr;
        float* d_currents = nullptr;
        int* d_spike_flags = nullptr;
        int* d_prefix_sum = nullptr;
        int* d_active_ids = nullptr;
        int* d_active_count = nullptr;
        float* d_x_pre = nullptr;
        float* d_x_post = nullptr;
        snn::SynapticEventRecord* d_events = nullptr;
        int* d_event_count = nullptr;

        const int max_events = static_cast<int>(connection_count * 2);

        try
        {
            CUDA_CHECK(cudaMalloc(&d_states, neuron_count * sizeof(snn::NeuronState)));
            CUDA_CHECK(cudaMalloc(&d_params, neuron_count * sizeof(snn::NeuronParameters)));
            CUDA_CHECK(cudaMalloc(&d_currents, neuron_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_spike_flags, neuron_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_prefix_sum, neuron_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_active_ids, neuron_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_x_pre, neuron_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_x_post, neuron_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_events, max_events * sizeof(snn::SynapticEventRecord)));
            CUDA_CHECK(cudaMalloc(&d_event_count, sizeof(int)));

            CUDA_CHECK(cudaMemcpy(d_states, std::vector<snn::NeuronState>(neuron_count, {-65.0f, -13.0f}).data(),
                                  neuron_count * sizeof(snn::NeuronState), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_params, params.data(), neuron_count * sizeof(snn::NeuronParameters), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_currents, input_currents.data(), neuron_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_x_pre, x_pre.data(), neuron_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_x_post, x_post.data(), neuron_count * sizeof(float), cudaMemcpyHostToDevice));

            // Run M2 neuron update + deterministic spike compaction
            const int blocks = static_cast<int>((neuron_count + threads_per_block - 1) / threads_per_block);
            snn::neuronUpdateKernel<<<blocks, threads_per_block>>>(
                d_states, d_params, d_currents, d_spike_flags, neuron_count, time_step);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(snn::compactSpikesInclusive(
                d_spike_flags, d_prefix_sum, d_active_ids, d_active_count,
                static_cast<int>(neuron_count), threads_per_block));
            CUDA_CHECK(cudaDeviceSynchronize());

            int gpu_active_count = 0;
            CUDA_CHECK(cudaMemcpy(&gpu_active_count, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));

            // Run M3 GPU CSR + CSC event-driven synaptic processing
            CUDA_CHECK(snn::processActiveSpikesGPU(
                dev_csr, dev_csc,
                d_active_ids, gpu_active_count,
                d_x_pre, d_x_post,
                a_plus, a_minus,
                d_events, d_event_count,
                max_events, threads_per_block));
            CUDA_CHECK(cudaDeviceSynchronize());

            int gpu_event_count = 0;
            CUDA_CHECK(cudaMemcpy(&gpu_event_count, d_event_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (gpu_event_count != static_cast<int>(expected_events.size()))
                throw std::runtime_error("GPU event count does not match CPU reference.");

            std::vector<snn::SynapticEventRecord> gpu_events(gpu_event_count);
            CUDA_CHECK(cudaMemcpy(gpu_events.data(), d_events,
                                  gpu_event_count * sizeof(snn::SynapticEventRecord),
                                  cudaMemcpyDeviceToHost));

            // 4. Order-independent validation
            std::sort(expected_events.begin(), expected_events.end(), eventLess);
            std::sort(gpu_events.begin(), gpu_events.end(), eventLess);

            std::vector<float> gpu_accumulated_currents(neuron_count, 0.0f);

            for (std::size_t i = 0; i < expected_events.size(); ++i)
            {
                const auto& e = expected_events[i];
                const auto& g = gpu_events[i];

                if (e.event_type != g.event_type ||
                    e.synapse_id != g.synapse_id ||
                    e.source != g.source ||
                    e.target != g.target)
                {
                    throw std::runtime_error("Event metadata/synapse_id mismatch.");
                }

                if (std::fabs(e.current_contribution - g.current_contribution) > 1e-5f ||
                    std::fabs(e.stdp_delta - g.stdp_delta) > 1e-5f)
                {
                    throw std::runtime_error("Event numerical contribution mismatch.");
                }

                if (g.event_type == static_cast<int>(snn::SynapticEventType::PresynapticCSR))
                    gpu_accumulated_currents[g.target] += g.current_contribution;
            }

            for (std::size_t i = 0; i < neuron_count; ++i)
            {
                if (std::fabs(gpu_accumulated_currents[i] - expected_currents[i]) > 1e-3f)
                    throw std::runtime_error("Accumulated postsynaptic current mismatch.");
            }

            std::cout << "M2 -> M3 GPU Integration & Synaptic Processing Test\n";
            std::cout << "Active spikes: " << gpu_active_count << '\n';
            std::cout << "Total CSR + CSC synaptic events: " << gpu_event_count << '\n';
            std::cout << "CSR presynaptic current & depression events: PASS\n";
            std::cout << "CSC postsynaptic potentiation & csc_to_csr mapping: PASS\n";
            std::cout << "OVERALL M3 GPU RESULT: PASS\n";

            cudaFree(d_states);
            cudaFree(d_params);
            cudaFree(d_currents);
            cudaFree(d_spike_flags);
            cudaFree(d_prefix_sum);
            cudaFree(d_active_ids);
            cudaFree(d_active_count);
            cudaFree(d_x_pre);
            cudaFree(d_x_post);
            cudaFree(d_events);
            cudaFree(d_event_count);
            snn::freeDeviceCSR(dev_csr);
            snn::freeDeviceCSC(dev_csc);
        }
        catch (...)
        {
            if (d_states) cudaFree(d_states);
            if (d_params) cudaFree(d_params);
            if (d_currents) cudaFree(d_currents);
            if (d_spike_flags) cudaFree(d_spike_flags);
            if (d_prefix_sum) cudaFree(d_prefix_sum);
            if (d_active_ids) cudaFree(d_active_ids);
            if (d_active_count) cudaFree(d_active_count);
            if (d_x_pre) cudaFree(d_x_pre);
            if (d_x_post) cudaFree(d_x_post);
            if (d_events) cudaFree(d_events);
            if (d_event_count) cudaFree(d_event_count);
            snn::freeDeviceCSR(dev_csr);
            snn::freeDeviceCSC(dev_csc);
            throw;
        }
    }
    catch (const std::exception& err)
    {
        std::cerr << "M3 GPU TEST FAILED: " << err.what() << '\n';
        return 1;
    }

    return 0;
}