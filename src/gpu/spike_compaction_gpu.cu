#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstddef>

namespace snn
{
    // Scatter active neuron IDs using inclusive prefix sums.
    __global__ void scatterActiveNeuronsKernel(
        const int* spike_flags,
        const int* prefix_sum,
        int* active_spike_ids,
        int neuron_count)
    {
        const int neuron_id =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (neuron_id >= neuron_count)
            return;

        if (spike_flags[neuron_id] != 0)
        {
            const int position =
                prefix_sum[neuron_id] - 1;

            active_spike_ids[position] = neuron_id;
        }
    }

    // Parallel inclusive prefix-sum and deterministic compaction.
    cudaError_t compactSpikesInclusive(
        const int* d_spike_flags,
        int* d_prefix_sum,
        int* d_active_spike_ids,
        int* d_active_spike_count,
        int neuron_count,
        int threads_per_block)
    {
        if (d_active_spike_count == nullptr)
            return cudaErrorInvalidValue;

        if (neuron_count < 0 || threads_per_block <= 0)
            return cudaErrorInvalidValue;

        if (neuron_count == 0)
        {
            return cudaMemset(
                d_active_spike_count,
                0,
                sizeof(int));
        }

        if (d_spike_flags == nullptr ||
            d_prefix_sum == nullptr ||
            d_active_spike_ids == nullptr)
        {
            return cudaErrorInvalidValue;
        }

        // Determine temporary storage required by CUB.
        void* d_temp_storage = nullptr;
        std::size_t temp_storage_bytes = 0;

        cudaError_t error =
            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_spike_flags,
                d_prefix_sum,
                neuron_count);

        if (error != cudaSuccess)
            return error;

        // Allocate temporary GPU storage.
        error = cudaMalloc(
            &d_temp_storage,
            temp_storage_bytes);

        if (error != cudaSuccess)
            return error;

        // Perform parallel inclusive prefix sum.
        error = cub::DeviceScan::InclusiveSum(
            d_temp_storage,
            temp_storage_bytes,
            d_spike_flags,
            d_prefix_sum,
            neuron_count);

        if (error != cudaSuccess)
        {
            cudaFree(d_temp_storage);
            return error;
        }

        // Final prefix-sum value = total active spikes.
        error = cudaMemcpy(
            d_active_spike_count,
            d_prefix_sum + neuron_count - 1,
            sizeof(int),
            cudaMemcpyDeviceToDevice);

        if (error != cudaSuccess)
        {
            cudaFree(d_temp_storage);
            return error;
        }

        // Configurable CUDA scatter launch.
        const int blocks =
            (neuron_count + threads_per_block - 1)
            / threads_per_block;

        scatterActiveNeuronsKernel<<<blocks, threads_per_block>>>(
            d_spike_flags,
            d_prefix_sum,
            d_active_spike_ids,
            neuron_count);

        error = cudaGetLastError();

        cudaFree(d_temp_storage);

        return error;
    }
}