#ifndef STDP_GPU_HPP
#define STDP_GPU_HPP

#include <cuda_runtime.h>
#include "synaptic_events.hpp"

namespace snn
{
    struct STDPParams
    {
        float a_plus = 0.01f;    // Potentiation amplitude (LTP)
        float a_minus = 0.012f;  // Depression amplitude (LTD)
        float tau_plus = 20.0f;  // Presynaptic trace time constant (ms)
        float tau_minus = 20.0f; // Postsynaptic trace time constant (ms)
        float w_min = 0.0f;      // Minimum synaptic weight bound
        float w_max = 1.0f;      // Maximum synaptic weight bound
    };

    // 1. Exponentially decay pre- and postsynaptic eligibility traces over timestep dt
    cudaError_t decaySTDPTracesGPU(
        float* d_x_pre,
        float* d_x_post,
        int neuron_count,
        float dt,
        float tau_plus,
        float tau_minus,
        int threads_per_block);

    // 2. Increment pre- and postsynaptic traces for active spiking neurons
    //    (Must execute AFTER processActiveSpikesGPU reads pre-increment traces)
    cudaError_t incrementSTDPTracesGPU(
        float* d_x_pre,
        float* d_x_post,
        const int* d_active_spike_ids,
        int active_spike_count,
        float pre_increment,
        float post_increment,
        int threads_per_block);

    // 3. Apply event-driven STDP weight updates to shared CSR weights and clamp to [w_min, w_max]
    cudaError_t applySTDPWeightUpdatesGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_weights,
        int connection_count,
        float w_min,
        float w_max,
        int threads_per_block);
}

#endif