#ifndef CURRENT_ACCUMULATION_HPP
#define CURRENT_ACCUMULATION_HPP

#include <cuda_runtime.h>
#include "synaptic_events.hpp"

namespace snn
{
    enum class CurrentAccumulationStrategy : int
    {
        Atomic = 0,
        BlockAggregated = 1,
        Gather = 2
    };

    struct CurrentContentionMetrics
    {
        int total_current_events = 0;
        int unique_targets_updated = 0;
        int max_updates_per_target = 0;
        float contention_ratio_ci = 0.0f; // total_current_events / unique_targets_updated
    };

    // Measures current contention C_I and hottest target count from presynaptic events
    cudaError_t measureCurrentContentionGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        int neuron_count,
        int* d_target_hit_counts,
        CurrentContentionMetrics& out_metrics,
        int threads_per_block);

    // Strategy 1: Direct global atomicAdd per presynaptic event
    cudaError_t accumulateCurrentsAtomicGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int neuron_count,
        int threads_per_block);

    // Strategy 2: Block-level shared-memory / warp-level aggregation before global atomicAdd
    cudaError_t accumulateCurrentsAggregatedGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int neuron_count,
        int threads_per_block);

    // Strategy 3: Atomic-free deterministic gather over incoming CSC edges
    cudaError_t accumulateCurrentsGatherGPU(
        const DeviceCSR& dev_csr,
        const DeviceCSC& dev_csc,
        const int* d_spike_flags,
        float* d_synaptic_currents,
        int threads_per_block);

    // Unified dispatcher for M4 & M7 adaptive selection
    cudaError_t accumulatePostsynapticCurrentsGPU(
        CurrentAccumulationStrategy strategy,
        const DeviceCSR& dev_csr,
        const DeviceCSC& dev_csc,
        const int* d_spike_flags,
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int threads_per_block);
}

#endif