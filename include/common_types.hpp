#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace snn
{
    // Unique identifier for each neuron
    using NeuronId = std::size_t;

    // Simulation time-step identifier
    using TimeStep = std::size_t;

    // State variables of an Izhikevich neuron
    struct NeuronState
    {
        float voltage;
        float recovery;
    };

    // Parameters of the Izhikevich neuron model
    struct NeuronParameters
    {
        float a;
        float b;
        float c;
        float d;
    };

    // Simulation configuration
    struct SimulationConfig
    {
        std::size_t neuron_count;
        std::size_t simulation_steps;
        float time_step;
    };

    // Information about an individual spike
    struct SpikeEvent
    {
        NeuronId neuron_id;
        TimeStep time_step;
    };

}

#endif