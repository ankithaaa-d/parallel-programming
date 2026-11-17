#ifndef NEURON_MODEL_HPP
#define NEURON_MODEL_HPP

#include "common_types.hpp"

namespace snn
{
    // Calculates the membrane potential derivative
    float calculateVoltageDerivative(
        const NeuronState& state,
        float input_current
    );

    // Calculates the recovery variable derivative
    float calculateRecoveryDerivative(
        const NeuronState& state,
        const NeuronParameters& parameters
    );

    // Updates the neuron state using the Izhikevich model
    bool updateNeuron(
        NeuronState& state,
        const NeuronParameters& parameters,
        float input_current,
        float time_step
    );
}

#endif