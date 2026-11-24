
#ifndef NETWORK_SIMULATOR_HPP
#define NETWORK_SIMULATOR_HPP

#include "common_types.hpp"

#include <vector>

namespace snn
{
    std::vector<SpikeEvent> simulateNetwork(
        std::vector<NeuronState>& states,
        const std::vector<NeuronParameters>& parameters,
        const std::vector<std::vector<float>>& input_currents,
        const SimulationConfig& config
    );
}

#endif
