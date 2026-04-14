// Copyright © 2023-2024 Apple Inc.

#include <memory>
#include <chrono>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

// Timeout for each command buffer before forcing a flush (in seconds).
// The macOS Metal GPU watchdog typically fires after ~5-10 seconds.
// We use a shorter timeout to be safe.
static const double kCommandBufferFlushIntervalSec = 2.0;

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

// Check if we should flush the current command buffer and start a new one
// to prevent GPU watchdog timeout during long-running operations.
void periodic_command_buffer_flush(Stream s) {
  static thread_local std::chrono::high_resolution_clock::time_point last_flush_time =
      std::chrono::high_resolution_clock::now();

  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(
      now - last_flush_time).count();

  if (elapsed >= kCommandBufferFlushIntervalSec) {
    auto& encoder = metal::get_command_encoder(s);
    auto* command_buffer = encoder.get_command_buffer();

    // Only flush if there's pending work
    if (encoder.needs_commit()) {
      // End current encoding
      encoder.end_encoding();

      // Commit the current command buffer with completion handler
      command_buffer->addCompletedHandler([](MTL::CommandBuffer* cbuf) {
        check_error(cbuf);
      });
      encoder.commit();

      // Update flush time
      last_flush_time = now;

      // Note: The next call to the encoder will automatically create
      // a new command buffer and compute encoder via needs_commit() check
    }
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);

    // Periodic flush to prevent GPU watchdog timeout
    // This splits long forward passes into chunks of kCommandBufferFlushIntervalSec seconds
    periodic_command_buffer_flush(s);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error(cbuf);
        });
    encoder.commit();
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { check_error(cbuf); });
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
}

} // namespace mlx::core::gpu
