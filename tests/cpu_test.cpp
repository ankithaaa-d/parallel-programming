#include "neuron_model.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

float readInput(const char* value, const std::string& name)
{
    std::size_t processed = 0;
    std::string input(value);

    float result = std::stof(input, &processed);

    if (processed != input.size() || !std::isfinite(result))
    {
        throw std::invalid_argument("Invalid input: " + name);
    }

    return result;
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 9)
        {
            std::cerr
                << "Usage: cpu_test <a> <b> <c> <d> "
                << "<initial_voltage> <initial_recovery> "
                << "<input_current> <time_step>\n";

            return 1;
        }

        snn::NeuronParameters parameters
        {
            readInput(argv[1], "a"),
            readInput(argv[2], "b"),
            readInput(argv[3], "c"),
            readInput(argv[4], "d")
        };

        snn::NeuronState state
        {
            readInput(argv[5], "initial_voltage"),
            readInput(argv[6], "initial_recovery")
        };

        const float input_current =
            readInput(argv[7], "input_current");

        const float time_step =
            readInput(argv[8], "time_step");

        const bool spike = snn::updateNeuron(
            state,
            parameters,
            input_current,
            time_step
        );

        std::cout << "Updated neuron state\n";
        std::cout << "Voltage: " << state.voltage << '\n';
        std::cout << "Recovery: " << state.recovery << '\n';
        std::cout << "Spike detected: "
                  << (spike ? "Yes" : "No") << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}