#include "synaptic_events.hpp"

#include <climits>
#include <stdexcept>
#include <vector>

namespace snn
{
    cudaError_t uploadCSRToDevice(const CSRConnectivity& host_csr, DeviceCSR& dev_csr)
    {
        freeDeviceCSR(dev_csr);

        const std::size_t neurons = host_csr.neuron_count();
        const std::size_t edges = host_csr.connection_count();

        if (neurons > static_cast<std::size_t>(INT_MAX) ||
            edges > static_cast<std::size_t>(INT_MAX))
        {
            return cudaErrorInvalidValue;
        }

        dev_csr.neuron_count = static_cast<int>(neurons);
        dev_csr.connection_count = static_cast<int>(edges);

        std::vector<int> row_offsets_i32(host_csr.row_offsets.size());
        for (std::size_t i = 0; i < host_csr.row_offsets.size(); ++i)
            row_offsets_i32[i] = static_cast<int>(host_csr.row_offsets[i]);

        std::vector<int> col_indices_i32(edges);
        for (std::size_t i = 0; i < edges; ++i)
            col_indices_i32[i] = static_cast<int>(host_csr.column_indices[i]);

        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&dev_csr.d_row_offsets),
            row_offsets_i32.size() * sizeof(int));
        if (err != cudaSuccess)
        {
            freeDeviceCSR(dev_csr);
            return err;
        }

        err = cudaMemcpy(
            dev_csr.d_row_offsets,
            row_offsets_i32.data(),
            row_offsets_i32.size() * sizeof(int),
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            freeDeviceCSR(dev_csr);
            return err;
        }

        if (edges > 0)
        {
            err = cudaMalloc(
                reinterpret_cast<void**>(&dev_csr.d_column_indices),
                edges * sizeof(int));
            if (err != cudaSuccess)
            {
                freeDeviceCSR(dev_csr);
                return err;
            }

            err = cudaMalloc(
                reinterpret_cast<void**>(&dev_csr.d_weights),
                edges * sizeof(float));
            if (err != cudaSuccess)
            {
                freeDeviceCSR(dev_csr);
                return err;
            }

            err = cudaMemcpy(
                dev_csr.d_column_indices,
                col_indices_i32.data(),
                edges * sizeof(int),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                freeDeviceCSR(dev_csr);
                return err;
            }

            err = cudaMemcpy(
                dev_csr.d_weights,
                host_csr.weights.data(),
                edges * sizeof(float),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                freeDeviceCSR(dev_csr);
                return err;
            }
        }

        return cudaSuccess;
    }

    void freeDeviceCSR(DeviceCSR& dev_csr)
    {
        if (dev_csr.d_row_offsets)
        {
            cudaFree(dev_csr.d_row_offsets);
            dev_csr.d_row_offsets = nullptr;
        }
        if (dev_csr.d_column_indices)
        {
            cudaFree(dev_csr.d_column_indices);
            dev_csr.d_column_indices = nullptr;
        }
        if (dev_csr.d_weights)
        {
            cudaFree(dev_csr.d_weights);
            dev_csr.d_weights = nullptr;
        }
        dev_csr.neuron_count = 0;
        dev_csr.connection_count = 0;
    }

    cudaError_t uploadCSCToDevice(const CSCConnectivity& host_csc, DeviceCSC& dev_csc)
    {
        freeDeviceCSC(dev_csc);

        const std::size_t neurons = host_csc.neuron_count();
        const std::size_t edges = host_csc.connection_count();

        if (neurons > static_cast<std::size_t>(INT_MAX) ||
            edges > static_cast<std::size_t>(INT_MAX) ||
            host_csc.csc_to_csr.size() != edges)
        {
            return cudaErrorInvalidValue;
        }

        dev_csc.neuron_count = static_cast<int>(neurons);
        dev_csc.connection_count = static_cast<int>(edges);

        std::vector<int> col_offsets_i32(host_csc.column_offsets.size());
        for (std::size_t i = 0; i < host_csc.column_offsets.size(); ++i)
            col_offsets_i32[i] = static_cast<int>(host_csc.column_offsets[i]);

        std::vector<int> row_indices_i32(edges);
        std::vector<int> csc_to_csr_i32(edges);
        for (std::size_t i = 0; i < edges; ++i)
        {
            row_indices_i32[i] = static_cast<int>(host_csc.row_indices[i]);
            csc_to_csr_i32[i] = static_cast<int>(host_csc.csc_to_csr[i]);
        }

        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&dev_csc.d_column_offsets),
            col_offsets_i32.size() * sizeof(int));
        if (err != cudaSuccess)
        {
            freeDeviceCSC(dev_csc);
            return err;
        }

        err = cudaMemcpy(
            dev_csc.d_column_offsets,
            col_offsets_i32.data(),
            col_offsets_i32.size() * sizeof(int),
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            freeDeviceCSC(dev_csc);
            return err;
        }

        if (edges > 0)
        {
            err = cudaMalloc(
                reinterpret_cast<void**>(&dev_csc.d_row_indices),
                edges * sizeof(int));
            if (err != cudaSuccess)
            {
                freeDeviceCSC(dev_csc);
                return err;
            }

            err = cudaMalloc(
                reinterpret_cast<void**>(&dev_csc.d_csc_to_csr),
                edges * sizeof(int));
            if (err != cudaSuccess)
            {
                freeDeviceCSC(dev_csc);
                return err;
            }

            err = cudaMemcpy(
                dev_csc.d_row_indices,
                row_indices_i32.data(),
                edges * sizeof(int),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                freeDeviceCSC(dev_csc);
                return err;
            }

            err = cudaMemcpy(
                dev_csc.d_csc_to_csr,
                csc_to_csr_i32.data(),
                edges * sizeof(int),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                freeDeviceCSC(dev_csc);
                return err;
            }
        }

        return cudaSuccess;
    }

    void freeDeviceCSC(DeviceCSC& dev_csc)
    {
        if (dev_csc.d_column_offsets)
        {
            cudaFree(dev_csc.d_column_offsets);
            dev_csc.d_column_offsets = nullptr;
        }
        if (dev_csc.d_row_indices)
        {
            cudaFree(dev_csc.d_row_indices);
            dev_csc.d_row_indices = nullptr;
        }
        if (dev_csc.d_csc_to_csr)
        {
            cudaFree(dev_csc.d_csc_to_csr);
            dev_csc.d_csc_to_csr = nullptr;
        }
        dev_csc.neuron_count = 0;
        dev_csc.connection_count = 0;
    }

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
        int max_events)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= active_spike_count)
            return;

        const int source = active_spike_ids[idx];
        const int start = row_offsets[source];
        const int end = row_offsets[source + 1];

        for (int edge = start; edge < end; ++edge)
        {
            const int target = column_indices[edge];
            const float weight = weights[edge];
            const float depression =
                (x_post != nullptr) ? (-a_minus * x_post[target]) : 0.0f;

            const int pos = atomicAdd(event_count, 1);
            if (pos < max_events)
            {
                events[pos] = SynapticEventRecord{
                    source,
                    target,
                    edge, // Authoritative CSR synapse ID
                    weight,
                    depression,
                    static_cast<int>(SynapticEventType::PresynapticCSR)
                };
            }
        }
    }

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
        int max_events)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= active_spike_count)
            return;

        const int target = active_spike_ids[idx];
        const int start = column_offsets[target];
        const int end = column_offsets[target + 1];

        for (int edge = start; edge < end; ++edge)
        {
            const int source = row_indices[edge];
            const int csr_synapse_id = csc_to_csr[edge];
            const float potentiation =
                (x_pre != nullptr) ? (a_plus * x_pre[source]) : 0.0f;

            const int pos = atomicAdd(event_count, 1);
            if (pos < max_events)
            {
                events[pos] = SynapticEventRecord{
                    source,
                    target,
                    csr_synapse_id, // Mapped to authoritative CSR synapse ID
                    0.0f,
                    potentiation,
                    static_cast<int>(SynapticEventType::PostsynapticCSC)
                };
            }
        }
    }

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
        int threads_per_block)
    {
        if (d_event_count == nullptr || threads_per_block <= 0 || max_events < 0)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaMemset(d_event_count, 0, sizeof(int));
        if (err != cudaSuccess)
            return err;

        if (active_spike_count <= 0 || dev_csr.connection_count == 0)
            return cudaSuccess;

        if (d_active_spike_ids == nullptr || (max_events > 0 && d_events == nullptr))
            return cudaErrorInvalidValue;

        const int blocks =
            (active_spike_count + threads_per_block - 1) / threads_per_block;

        // 1. Outgoing presynaptic CSR events (current propagation + STDP depression)
        presynapticCSREventsKernel<<<blocks, threads_per_block>>>(
            d_active_spike_ids,
            active_spike_count,
            dev_csr.d_row_offsets,
            dev_csr.d_column_indices,
            dev_csr.d_weights,
            d_x_post,
            a_minus,
            d_events,
            d_event_count,
            max_events);

        err = cudaGetLastError();
        if (err != cudaSuccess)
            return err;

        // 2. Incoming postsynaptic CSC events (STDP potentiation via csc_to_csr)
        postsynapticCSCEventsKernel<<<blocks, threads_per_block>>>(
            d_active_spike_ids,
            active_spike_count,
            dev_csc.d_column_offsets,
            dev_csc.d_row_indices,
            dev_csc.d_csc_to_csr,
            d_x_pre,
            a_plus,
            d_events,
            d_event_count,
            max_events);

        return cudaGetLastError();
    }
}