#include "stdp_gpu.hpp"

#include <cmath>

namespace snn
{
    namespace
    {
        __global__ void decaySTDPTracesKernel(
            float* x_pre,
            float* x_post,
            int neuron_count,
            float decay_pre,
            float decay_post)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= neuron_count)
                return;

            x_pre[idx] *= decay_pre;
            x_post[idx] *= decay_post;
        }

        __global__ void incrementSTDPTracesKernel(
            float* x_pre,
            float* x_post,
            const int* active_spike_ids,
            int active_spike_count,
            float pre_increment,
            float post_increment)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= active_spike_count)
                return;

            const int nid = active_spike_ids[idx];
            x_pre[nid] += pre_increment;
            x_post[nid] += post_increment;
        }

        __global__ void applySTDPDeltaKernel(
            const SynapticEventRecord* events,
            int event_count,
            float* weights,
            int connection_count)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= event_count)
                return;

            const SynapticEventRecord ev = events[idx];
            if (ev.stdp_delta != 0.0f &&
                ev.synapse_id >= 0 &&
                ev.synapse_id < connection_count)
            {
                atomicAdd(&weights[ev.synapse_id], ev.stdp_delta);
            }
        }

        __global__ void clampWeightsKernel(
            float* weights,
            int connection_count,
            float w_min,
            float w_max)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= connection_count)
                return;

            float w = weights[idx];
            if (w < w_min)
                w = w_min;
            else if (w > w_max)
                w = w_max;
            weights[idx] = w;
        }
    } // namespace

    cudaError_t decaySTDPTracesGPU(
        float* d_x_pre,
        float* d_x_post,
        int neuron_count,
        float dt,
        float tau_plus,
        float tau_minus,
        int threads_per_block)
    {
        if (neuron_count <= 0 || threads_per_block <= 0 ||
            d_x_pre == nullptr || d_x_post == nullptr ||
            dt < 0.0f || tau_plus <= 0.0f || tau_minus <= 0.0f)
        {
            return cudaErrorInvalidValue;
        }

        const float decay_pre = std::exp(-dt / tau_plus);
        const float decay_post = std::exp(-dt / tau_minus);

        const int blocks = (neuron_count + threads_per_block - 1) / threads_per_block;
        decaySTDPTracesKernel<<<blocks, threads_per_block>>>(
            d_x_pre, d_x_post, neuron_count, decay_pre, decay_post);

        return cudaGetLastError();
    }

    cudaError_t incrementSTDPTracesGPU(
        float* d_x_pre,
        float* d_x_post,
        const int* d_active_spike_ids,
        int active_spike_count,
        float pre_increment,
        float post_increment,
        int threads_per_block)
    {
        if (threads_per_block <= 0)
            return cudaErrorInvalidValue;

        if (active_spike_count <= 0)
            return cudaSuccess;

        if (d_x_pre == nullptr || d_x_post == nullptr || d_active_spike_ids == nullptr)
            return cudaErrorInvalidValue;

        const int blocks = (active_spike_count + threads_per_block - 1) / threads_per_block;
        incrementSTDPTracesKernel<<<blocks, threads_per_block>>>(
            d_x_pre,
            d_x_post,
            d_active_spike_ids,
            active_spike_count,
            pre_increment,
            post_increment);

        return cudaGetLastError();
    }

    cudaError_t applySTDPWeightUpdatesGPU(
        const SynapticEventRecord* d_events,
        int event_count,
        float* d_weights,
        int connection_count,
        float w_min,
        float w_max,
        int threads_per_block)
    {
        if (threads_per_block <= 0 || w_min > w_max)
            return cudaErrorInvalidValue;

        if (connection_count <= 0 || event_count <= 0)
            return cudaSuccess;

        if (d_events == nullptr || d_weights == nullptr)
            return cudaErrorInvalidValue;

        const int event_blocks = (event_count + threads_per_block - 1) / threads_per_block;
        applySTDPDeltaKernel<<<event_blocks, threads_per_block>>>(
            d_events, event_count, d_weights, connection_count);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;

        const int weight_blocks = (connection_count + threads_per_block - 1) / threads_per_block;
        clampWeightsKernel<<<weight_blocks, threads_per_block>>>(
            d_weights, connection_count, w_min, w_max);

        return cudaGetLastError();
    }
}