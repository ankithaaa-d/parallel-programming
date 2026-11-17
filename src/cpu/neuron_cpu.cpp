#include "neuron_model.hpp"

#include <cmath>
#include <stdexcept>

namespace snn
{
    namespace
    {
        // Constants intrinsic to the Izhikevich model
        constexpr float quadratic_coefficient = 0.04f;
        constexpr float linear_coefficient = 5.0f;
        constexpr float constant_term = 140.0f;
        constexpr float spike_threshold = 30.0f;
    }

    float calculateVoltageDerivative(
        const NeuronState& state,
        float input_current)
    {
        return quadratic_coefficient * state.voltage * state.voltage
             + linear_coefficient * state.voltage
             + constant_term
             - state.recovery
             + input_current;
    }

    float calculateRecoveryDerivative(
        const NeuronState& state,
        const NeuronParameters& parameters)
    {
        return parameters.a *
               (parameters.b * state.voltage - state.recovery);
    }

    bool updateNeuron(
        NeuronState& state,
        const NeuronParameters& parameters,
        float input_current,
        float time_step)
    {
        if (!std::isfinite(time_step) || time_step <= 0.0f)
        {
            throw std::invalid_argument("Time step must be positive and finite.");
        }

        if (!std::isfinite(input_current))
        {
            throw std::invalid_argument("Input current must be finite.");
        }

        // First half-step for membrane potential
        state.voltage += 0.5f * time_step *
            calculateVoltageDerivative(state, input_current);

        // Second half-step using the updated voltage
        state.voltage += 0.5f * time_step *
            calculateVoltageDerivative(state, input_current);

        // Update recovery variable
        state.recovery += time_step *
            calculateRecoveryDerivative(state, parameters);

        // Spike detection and reset
        if (state.voltage >= spike_threshold)
        {
            state.voltage = parameters.c;
            state.recovery += parameters.d;

            return true;
        }

        return false;
    }
}