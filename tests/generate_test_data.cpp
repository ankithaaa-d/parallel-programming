#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr
            << "Usage: generate_test_data <neuron_count> <seed> <output_file>\n";
        return 1;
    }

    try
    {
        std::size_t processed = 0;
        const std::string count_arg = argv[1];
        const auto neuron_count = std::stoull(count_arg, &processed);

        if (processed != count_arg.size() || neuron_count == 0)
            throw std::invalid_argument("Invalid neuron count.");

        const std::string seed_arg = argv[2];
        const auto seed = std::stoull(seed_arg, &processed);

        if (processed != seed_arg.size())
            throw std::invalid_argument("Invalid seed.");

        const std::string output_path = argv[3];

        std::mt19937_64 rng(seed);

        // Parameter distributions for valid Izhikevich neuron models.
        std::uniform_real_distribution<float> a_dist(0.02f, 0.1f);
        std::uniform_real_distribution<float> b_dist(0.2f, 0.25f);
        std::uniform_real_distribution<float> c_dist(-65.0f, -50.0f);
        std::uniform_real_distribution<float> d_dist(2.0f, 8.0f);
        std::uniform_real_distribution<float> voltage_dist(-70.0f, -55.0f);
        std::uniform_real_distribution<float> recovery_dist(-18.0f, -10.0f);
        std::uniform_real_distribution<float> current_dist(0.0f, 200.0f);

        constexpr float time_step = 1.0f;

        std::ofstream output(output_path);

        if (!output)
            throw std::runtime_error("Cannot create output file.");

        output.precision(9);
        output << neuron_count << ' ' << time_step << '\n';

        for (std::size_t i = 0; i < neuron_count; ++i)
        {
            output
                << a_dist(rng) << ' '
                << b_dist(rng) << ' '
                << c_dist(rng) << ' '
                << d_dist(rng) << ' '
                << voltage_dist(rng) << ' '
                << recovery_dist(rng) << ' '
                << current_dist(rng) << '\n';
        }

        output.close();

        if (!output)
            throw std::runtime_error("Failed to write input file.");

        std::cout << "M2 input generated successfully.\n"
                  << "Neurons: " << neuron_count << '\n'
                  << "Seed: " << seed << '\n'
                  << "Output: " << output_path << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}