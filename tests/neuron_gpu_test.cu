#include <cuda_runtime.h>

#include "neuron_model.hpp"
#include "common_types.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
        {                                                     \
            throw std::runtime_error(cudaGetErrorString(error)); \
        }                                                     \
    } while (0)

namespace snn
{
    __global__ void neuronUpdateKernel(
        NeuronState* states,
        const NeuronParameters* parameters,
        const float* input_currents,
        int* spike_flags,
        std::size_t neuron_count,
        float time_step);
}

float readInput(const char* value, const std::string& name)
{
    std::size_t processed = 0;
    std::string input(value);

    float result = std::stof(input, &processed);

    if (processed != input.size() || !std::isfinite(result))
        throw std::invalid_argument("Invalid input: " + name);

    return result;
}

int main(int argc, char* argv[])
{
    if (argc != 9)
    {
        std::cerr
            << "Usage: neuron_gpu_test <a> <b> <c> <d> "
            << "<initial_voltage> <initial_recovery> "
            << "<input_current> <time_step>\n";

        return 1;
    }

    try
    {
        // Read runtime inputs
        snn::NeuronParameters parameters{
            readInput(argv[1], "a"),
            readInput(argv[2], "b"),
            readInput(argv[3], "c"),
            readInput(argv[4], "d")
        };

        snn::NeuronState initial_state{
            readInput(argv[5], "initial_voltage"),
            readInput(argv[6], "initial_recovery")
        };

        const float input_current =
            readInput(argv[7], "input_current");

        const float time_step =
            readInput(argv[8], "time_step");

        if (time_step <= 0.0f)
            throw std::invalid_argument(
                "Time step must be positive.");

        // CPU reference
        snn::NeuronState cpu_state = initial_state;

        const bool cpu_spike = snn::updateNeuron(
            cpu_state,
            parameters,
            input_current,
            time_step
        );

        // GPU input
        snn::NeuronState gpu_state = initial_state;
        int gpu_spike = 0;

        snn::NeuronState* d_state = nullptr;
        snn::NeuronParameters* d_parameters = nullptr;
        float* d_current = nullptr;
        int* d_spike = nullptr;

        try
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_state),
                sizeof(snn::NeuronState)));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_parameters),
                sizeof(snn::NeuronParameters)));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_current),
                sizeof(float)));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_spike),
                sizeof(int)));

            CUDA_CHECK(cudaMemcpy(
                d_state,
                &gpu_state,
                sizeof(snn::NeuronState),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_parameters,
                &parameters,
                sizeof(snn::NeuronParameters),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_current,
                &input_current,
                sizeof(float),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemset(d_spike, 0, sizeof(int)));

            snn::neuronUpdateKernel<<<1, 1>>>(
                d_state,
                d_parameters,
                d_current,
                d_spike,
                1,
                time_step
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(
                &gpu_state,
                d_state,
                sizeof(snn::NeuronState),
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                &gpu_spike,
                d_spike,
                sizeof(int),
                cudaMemcpyDeviceToHost));
        }
        catch (...)
        {
            if (d_state) cudaFree(d_state);
            if (d_parameters) cudaFree(d_parameters);
            if (d_current) cudaFree(d_current);
            if (d_spike) cudaFree(d_spike);
            throw;
        }

        CUDA_CHECK(cudaFree(d_state));
        CUDA_CHECK(cudaFree(d_parameters));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_spike));

        // Compare CPU and GPU
        constexpr float tolerance = 1e-4f;

        const bool voltage_match =
            std::fabs(cpu_state.voltage - gpu_state.voltage)
            <= tolerance;

        const bool recovery_match =
            std::fabs(cpu_state.recovery - gpu_state.recovery)
            <= tolerance;

        const bool spike_match =
            cpu_spike == (gpu_spike != 0);

        std::cout << "\nCPU results\n";
        std::cout << "Voltage: " << cpu_state.voltage << '\n';
        std::cout << "Recovery: " << cpu_state.recovery << '\n';
        std::cout << "Spike: " << cpu_spike << '\n';

        std::cout << "\nGPU results\n";
        std::cout << "Voltage: " << gpu_state.voltage << '\n';
        std::cout << "Recovery: " << gpu_state.recovery << '\n';
        std::cout << "Spike: " << gpu_spike << '\n';

        std::cout << "\nComparison\n";
        std::cout << "Voltage match: "
                  << (voltage_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Recovery match: "
                  << (recovery_match ? "PASS" : "FAIL") << '\n';

        std::cout << "Spike match: "
                  << (spike_match ? "PASS" : "FAIL") << '\n';

        if (voltage_match && recovery_match && spike_match)
        {
            std::cout << "\nCPU and GPU results MATCH.\n";
        }
        else
        {
            std::cout << "\nCPU and GPU results DO NOT MATCH.\n";
            return 1;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}