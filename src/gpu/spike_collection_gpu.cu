#include <cuda_runtime.h>
#include <cstddef>

namespace snn
{
    __global__ void atomicSpikeCollectionKernel(
        const int* spike_flags,
        int* active_spike_ids,
        int* active_spike_count,
        std::size_t neuron_count)
    {
        const std::size_t neuron_id =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (neuron_id >= neuron_count)
            return;

        if (spike_flags[neuron_id] != 0)
        {
            const int position = atomicAdd(
                active_spike_count, 1);

            active_spike_ids[position] =
                static_cast<int>(neuron_id);
        }
    }
}