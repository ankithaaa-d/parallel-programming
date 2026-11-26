#include "current_accumulation.hpp"

#include <algorithm>
#include <vector>

namespace snn
{
    namespace
    {
        constexpr int kMaxSharedNeurons = 4096; // 16 KB shared memory per block

        __global__ void countTargetHitsKernel(
            const SynapticEventRecord* events,
            int event_count,
            int* target_hit_counts)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= event_count)
                return;

            const SynapticEventRecord ev = events[idx];
            if (ev.event_type == static_cast<int>(SynapticEventType::PresynapticCSR))
            {
                atomicAdd(&target_hit_counts[ev.target], 1);
            }
        }

        __global__ void atomicCurrentAccumulationKernel(
            const SynapticEventRecord* events,
            int event_count,
            float* synaptic_currents)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= event_count)
                return;

            const SynapticEventRecord ev = events[idx];
            if (ev.event_type == static_cast<int>(SynapticEventType::PresynapticCSR))
            {
                atomicAdd(&synaptic_currents[ev.target], ev.current_contribution);
            }
        }

        // Shared-memory block-level aggregation for networks up to kMaxSharedNeurons
        __global__ void sharedBlockAggregatedCurrentKernel(
            const SynapticEventRecord* events,
            int event_count,
            float* synaptic_currents,
            int neuron_count)
        {
            extern __shared__ float shared_currents[];

            for (int n = threadIdx.x; n < neuron_count; n += blockDim.x)
                shared_currents[n] = 0.0f;

            __syncthreads();

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < event_count)
            {
                const SynapticEventRecord ev = events[idx];
                if (ev.event_type == static_cast<int>(SynapticEventType::PresynapticCSR))
                {
                    atomicAdd(&shared_currents[ev.target], ev.current_contribution);
                }
            }

            __syncthreads();

            for (int n = threadIdx.x; n < neuron_count; n += blockDim.x)
            {
                const float val = shared_currents[n];
                if (val != 0.0f)
                    atomicAdd(&synaptic_currents[n], val);
            }
        }

        // Warp-level peer aggregation for arbitrary neuron_count (> kMaxSharedNeurons)
        __global__ void warpAggregatedCurrentKernel(
            const SynapticEventRecord* events,
            int event_count,
            float* synaptic_currents)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int active_mask = __activemask();

            bool valid = false;
            int target = -1;
            float contrib = 0.0f;

            if (idx < event_count)
            {
                const SynapticEventRecord ev = events[idx];
                if (ev.event_type == static_cast<int>(SynapticEventType::PresynapticCSR))
                {
                    valid = true;
                    target = ev.target;
                    contrib = ev.current_contribution;
                }
            }

            if (!valid)
                return;

            const unsigned int peers = __match_any_sync(active_mask, target);
            const int lane_id = threadIdx.x & 31;
            const int leader = __ffs(peers) - 1;

            float sum = contrib;
            unsigned int remaining = peers & ~(1u << leader);
            while (remaining != 0)
            {
                const int peer_lane = __ffs(remaining) - 1;
                const float peer_val = __shfl_sync(peers, contrib, peer_lane);
                if (lane_id == leader)
                    sum += peer_val;
                remaining &= ~(1u << peer_lane);
            }

            if (lane_id == leader)
                atomicAdd(&synaptic_currents[target], sum);
        }

        // Gather-based accumulation: one thread per postsynaptic neuron over CSC
        __global__ void gatherCurrentAccumulationKernel(
            int neuron_count,
            const int* column_offsets,
            const int* row_indices,
            const int* csc_to_csr,
            const float* weights,
            const int* spike_flags,
            float* synaptic_currents)
        {
            const int target = blockIdx.x * blockDim.x + threadIdx.x;
            if (target >= neuron_count)
                return;

            const int start = column_offsets[target];
            const int end = column_offsets[target + 1];

            float total = 0.0f;
            for (int edge = start; edge < end; ++edge)
            {
                const int source = row_indices[edge];
                if (spike_flags[source] != 0)
                {
                    const int csr_edge = csc_to_csr[edge];
                    total += weights[csr_edge];
                }
            }

            synaptic_currents[target] = total;
        }
    } // namespace

    cudaError_t measureCurrentContentionGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        int neuron_count,
        int* d_target_hit_counts,
        CurrentContentionMetrics& out_metrics,
        int threads_per_block)
    {
        out_metrics = CurrentContentionMetrics{};

        if (neuron_count <= 0 || threads_per_block <= 0 || d_target_hit_counts == nullptr)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaMemset(
            d_target_hit_counts, 0, static_cast<std::size_t>(neuron_count) * sizeof(int));
        if (err != cudaSuccess)
            return err;

        if (event_count <= 0)
            return cudaSuccess;

        if (d_events == nullptr)
            return cudaErrorInvalidValue;

        const int blocks = (event_count + threads_per_block - 1) / threads_per_block;
        countTargetHitsKernel<<<blocks, threads_per_block>>>(
            d_events, event_count, d_target_hit_counts);

        err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;

        std::vector<int> host_hits(static_cast<std::size_t>(neuron_count), 0);
        err = cudaMemcpy(
            host_hits.data(),
            d_target_hit_counts,
            static_cast<std::size_t>(neuron_count) * sizeof(int),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return err;

        int total_events = 0;
        int unique_targets = 0;
        int max_hits = 0;

        for (int count : host_hits)
        {
            if (count > 0)
            {
                total_events += count;
                ++unique_targets;
                if (count > max_hits)
                    max_hits = count;
            }
        }

        out_metrics.total_current_events = total_events;
        out_metrics.unique_targets_updated = unique_targets;
        out_metrics.max_updates_per_target = max_hits;
        out_metrics.contention_ratio_ci =
            (unique_targets > 0)
                ? (static_cast<float>(total_events) / static_cast<float>(unique_targets))
                : 0.0f;

        return cudaSuccess;
    }

    cudaError_t accumulateCurrentsAtomicGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int neuron_count,
        int threads_per_block)
    {
        if (neuron_count <= 0 || threads_per_block <= 0 || d_synaptic_currents == nullptr)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaMemset(
            d_synaptic_currents, 0, static_cast<std::size_t>(neuron_count) * sizeof(float));
        if (err != cudaSuccess)
            return err;

        if (event_count <= 0)
            return cudaSuccess;

        if (d_events == nullptr)
            return cudaErrorInvalidValue;

        const int blocks = (event_count + threads_per_block - 1) / threads_per_block;
        atomicCurrentAccumulationKernel<<<blocks, threads_per_block>>>(
            d_events, event_count, d_synaptic_currents);

        return cudaGetLastError();
    }

    cudaError_t accumulateCurrentsAggregatedGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int neuron_count,
        int threads_per_block)
    {
        if (neuron_count <= 0 || threads_per_block <= 0 || d_synaptic_currents == nullptr)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaMemset(
            d_synaptic_currents, 0, static_cast<std::size_t>(neuron_count) * sizeof(float));
        if (err != cudaSuccess)
            return err;

        if (event_count <= 0)
            return cudaSuccess;

        if (d_events == nullptr)
            return cudaErrorInvalidValue;

        const int blocks = (event_count + threads_per_block - 1) / threads_per_block;

        if (neuron_count <= kMaxSharedNeurons)
        {
            const std::size_t shared_bytes =
                static_cast<std::size_t>(neuron_count) * sizeof(float);
            sharedBlockAggregatedCurrentKernel<<<blocks, threads_per_block, shared_bytes>>>(
                d_events, event_count, d_synaptic_currents, neuron_count);
        }
        else
        {
            warpAggregatedCurrentKernel<<<blocks, threads_per_block>>>(
                d_events, event_count, d_synaptic_currents);
        }

        return cudaGetLastError();
    }

    cudaError_t accumulateCurrentsGatherGPU(
        const DeviceCSR& dev_csr,
        const DeviceCSC& dev_csc,
        const int* d_spike_flags,
        float* d_synaptic_currents,
        int threads_per_block)
    {
        const int neuron_count = dev_csc.neuron_count;
        if (neuron_count <= 0 || threads_per_block <= 0 ||
            d_spike_flags == nullptr || d_synaptic_currents == nullptr)
        {
            return cudaErrorInvalidValue;
        }

        if (dev_csc.connection_count == 0)
        {
            return cudaMemset(
                d_synaptic_currents, 0, static_cast<std::size_t>(neuron_count) * sizeof(float));
        }

        const int blocks = (neuron_count + threads_per_block - 1) / threads_per_block;
        gatherCurrentAccumulationKernel<<<blocks, threads_per_block>>>(
            neuron_count,
            dev_csc.d_column_offsets,
            dev_csc.d_row_indices,
            dev_csc.d_csc_to_csr,
            dev_csr.d_weights,
            d_spike_flags,
            d_synaptic_currents);

        return cudaGetLastError();
    }

    cudaError_t accumulatePostsynapticCurrentsGPU(
        CurrentAccumulationStrategy strategy,
        const DeviceCSR& dev_csr,
        const DeviceCSC& dev_csc,
        const int* d_spike_flags,
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_synaptic_currents,
        int threads_per_block)
    {
        switch (strategy)
        {
            case CurrentAccumulationStrategy::Atomic:
                return accumulateCurrentsAtomicGPU(
                    d_events, event_count, d_synaptic_currents,
                    dev_csr.neuron_count, threads_per_block);

            case CurrentAccumulationStrategy::BlockAggregated:
                return accumulateCurrentsAggregatedGPU(
                    d_events, event_count, d_synaptic_currents,
                    dev_csr.neuron_count, threads_per_block);

            case CurrentAccumulationStrategy::Gather:
                return accumulateCurrentsGatherGPU(
                    dev_csr, dev_csc, d_spike_flags,
                    d_synaptic_currents, threads_per_block);

            default:
                return cudaErrorInvalidValue;
        }
    }
}