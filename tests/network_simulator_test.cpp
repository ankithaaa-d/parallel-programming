
#include "network_simulator.hpp"

#include <iostream>
#include <vector>

int main()
{
    std::vector<snn::NeuronState> states =
    {
        {-65.0f, -13.0f},
        {-65.0f, -13.0f}
    };

    std::vector<snn::NeuronParameters> parameters =
    {
        {0.02f, 0.2f, -65.0f, 8.0f},
        {0.02f, 0.2f, -65.0f, 8.0f}
    };

    // Each row represents one time step.
    // Each value represents the input current for one neuron.
    std::vector<std::vector<float>> input_currents =
    {
        {10.0f, 1000.0f},
        {10.0f, 0.0f},
        {10.0f, 0.0f}
    };

    snn::SimulationConfig config
    {
        2,      // Number of neurons
        3,      // Number of simulation steps
        1.0f    // Time step
    };

    const std::vector<snn::SpikeEvent> spikes =
        snn::simulateNetwork(
            states,
            parameters,
            input_currents,
            config
        );

    std::cout << "Sequential network simulation results\n";

    for (std::size_t i = 0; i < spikes.size(); ++i)
    {
        std::cout << "Spike: neuron "
                  << spikes[i].neuron_id
                  << " at time step "
                  << spikes[i].time_step << '\n';
    }

    std::cout << "\nFinal neuron states:\n";

    for (std::size_t i = 0; i < states.size(); ++i)
    {
        std::cout << "Neuron " << i
                  << " | Voltage: " << states[i].voltage
                  << " | Recovery: " << states[i].recovery
                  << '\n';
    }

    if (spikes.size() != 1 ||
        spikes[0].neuron_id != 1 ||
        spikes[0].time_step != 0)
    {
        std::cerr << "TEST FAILED: Unexpected spike events.\n";
        return 1;
    }

    std::cout << "\nTEST PASSED\n";
    return 0;
}
