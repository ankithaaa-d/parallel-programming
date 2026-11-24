#include <cuda_runtime.h>
#include "common_types.hpp"

namespace snn
{
    __global__ void neuronUpdateKernel(
        NeuronState* states,
        const NeuronParameters* parameters,
        const float* input_currents,
        int* spike_flags,
        std::size_t neuron_count,
        float time_step)
    {
        const std::size_t neuron_id =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (neuron_id >= neuron_count)
            return;

        NeuronState state = states[neuron_id];
        const NeuronParameters params = parameters[neuron_id];

        const float current = input_currents[neuron_id];

        constexpr float threshold = 30.0f;

        // First half-step for membrane potential
        float derivative =
            0.04f * state.voltage * state.voltage
            + 5.0f * state.voltage
            + 140.0f
            - state.recovery
            + current;

        state.voltage += 0.5f * time_step * derivative;

        // Second half-step
        derivative =
            0.04f * state.voltage * state.voltage
            + 5.0f * state.voltage
            + 140.0f
            - state.recovery
            + current;

        state.voltage += 0.5f * time_step * derivative;

        // Recovery update
        state.recovery += time_step * params.a *
            (params.b * state.voltage - state.recovery);

        // Spike detection and reset
        int spike = 0;

        if (state.voltage >= threshold)
        {
            state.voltage = params.c;
            state.recovery += params.d;
            spike = 1;
        }

        states[neuron_id] = state;
        spike_flags[neuron_id] = spike;
    }
}