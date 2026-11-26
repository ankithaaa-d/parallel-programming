#ifndef SYNAPTIC_EVENTS_HPP
#define SYNAPTIC_EVENTS_HPP

#include <cuda_runtime.h>
#include "connectivity.hpp"

namespace snn
{
    enum class SynapticEventType : int
    {
        PresynapticCSR = 0,
        PostsynapticCSC = 1
    };

    // Output record emitted by M3 event-driven GPU kernels
    // Consumed by M4 (current accumulation) and M5/M6 (STDP + weight updates)
    struct SynapticEventRecord
    {
        int source;
        int target;
        int synapse_id;             // Authoritative CSR index
        float current_contribution; // Non-zero for presynaptic CSR events
        float stdp_delta;           // Pre-increment trace STDP contribution
        int event_type;             // SynapticEventType
    };

    // Device-side CSR view (owns single shared weights array)
    struct DeviceCSR
    {
        int neuron_count = 0;
        int connection_count = 0;
        int* d_row_offsets = nullptr;
        int* d_column_indices = nullptr;
        float* d_weights = nullptr;
    };

    // Device-side CSC view (maps to shared CSR weights via d_csc_to_csr)
    struct DeviceCSC
    {
        int neuron_count = 0;
        int connection_count = 0;
        int* d_column_offsets = nullptr;
        int* d_row_indices = nullptr;
        int* d_csc_to_csr = nullptr;
    };

    // Device allocation & upload helpers
    cudaError_t uploadCSRToDevice(const CSRConnectivity& host_csr, DeviceCSR& dev_csr);
    void freeDeviceCSR(DeviceCSR& dev_csr);

    cudaError_t uploadCSCToDevice(const CSCConnectivity& host_csc, DeviceCSC& dev_csc);
    void freeDeviceCSC(DeviceCSC& dev_csc);

    // M3 GPU Kernels
    __global__ void presynapticCSREventsKernel(
        const int* active_spike_ids,
        int active_spike_count,
        const int* row_offsets,
        const int* column_indices,
        const float* weights,
        const float* x_post,
        float a_minus,
        SynapticEventRecord* events,
        int* event_count,
        int max_events);

    __global__ void postsynapticCSCEventsKernel(
        const int* active_spike_ids,
        int active_spike_count,
        const int* column_offsets,
        const int* row_indices,
        const int* csc_to_csr,
        const float* x_pre,
        float a_plus,
        SynapticEventRecord* events,
        int* event_count,
        int max_events);

    // Host launcher: processes active spikes through CSR (outgoing) and CSC (incoming)
    // Pass d_x_pre = nullptr / d_x_post = nullptr when running pure connectivity propagation
    cudaError_t processActiveSpikesGPU(
        const DeviceCSR& dev_csr,
        const DeviceCSC& dev_csc,
        const int* d_active_spike_ids,
        int active_spike_count,
        const float* d_x_pre,
        const float* d_x_post,
        float a_plus,
        float a_minus,
        SynapticEventRecord* d_events,
        int* d_event_count,
        int max_events,
        int threads_per_block);
}

#endif