
#include "network_simulator.hpp"
#include "neuron_model.hpp"

#include <stdexcept>

namespace snn
{
    std::vector<SpikeEvent> simulateNetwork(
        std::vector<NeuronState>& states,
        const std::vector<NeuronParameters>& parameters,
        const std::vector<std::vector<float>>& input_currents,
        const SimulationConfig& config)
    {
        if (states.size() != config.neuron_count ||
            parameters.size() != config.neuron_count)
        {
            throw std::invalid_argument(
                "Neuron states and parameters must match neuron count."
            );
        }

        if (input_currents.size() != config.simulation_steps)
        {
            throw std::invalid_argument(
                "Input currents must match simulation steps."
            );
        }

        std::vector<SpikeEvent> spike_events;

        for (TimeStep step = 0;
             step < config.simulation_steps;
             ++step)
        {
            if (input_currents[step].size() != config.neuron_count)
            {
                throw std::invalid_argument(
                    "Each time step must contain one current per neuron."
                );
            }

            for (NeuronId neuron = 0;
                 neuron < config.neuron_count;
                 ++neuron)
            {
                const bool spiked = updateNeuron(
                    states[neuron],
                    parameters[neuron],
                    input_currents[step][neuron],
                    config.time_step
                );

                if (spiked)
                {
                    spike_events.push_back(
                        {neuron, step}
                    );
                }
            }
        }

        return spike_events;
    }
}
