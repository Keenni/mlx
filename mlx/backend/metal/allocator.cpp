// Copyright © 2023-2024 Apple Inc.

#include \"mlx/backend/metal/allocator.h\"
#include \"mlx/backend/gpu/device_info.h\"
#include \"mlx/backend/metal/metal.h\"
#include \"mlx/backend/metal/resident.h\"
#include \"mlx/memory.h\"

#include <mach/vm_page_size.h>
#include <unistd.h>
#include <cstdlib>
#include <sys/mman.h>
#include <fcntl.h>
#include <random>
#include <cstdio>

namespace mlx::core {

constexpr size_t resource_options =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

namespace allocator {

Allocator& allocator() {
  return metal::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<MTL::Buffer*>(ptr_)->contents();
}

} // namespace allocator

namespace metal {

MetalAllocator::MetalAllocator(Device& d)
    : device_(d.mtl_device()),
      residency_set_(d.residency_set()),
      buffer_cache_(
          vm_page_size,
          [](MTL::Buffer* buf) { return buf->length(); },
          [this](MTL::Buffer* buf) {
            if (!buf->heap()) {
              residency_set_.erase(buf);
            }
            auto pool = metal::new_scoped_memory_pool();
            buf->release();
          }) {
  const auto& info = gpu::device_info(0);
  auto memsize = std::get<size_t>(info.at(\"memory_size\"));
  auto max_rec_size =
      std::get<size_t>(info.at(\"max_recommended_working_set_size\"));
  resource_limit_ = std::get<size_t>(info.at(\"resource_limit\"));
  block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);
  gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);
  max_pool_size_ = block_limit_;
  bool is_vm = std::get<std::string>(info.at(\"device_name\")) ==
      \"Apple Paravirtual device\";
  if (is_vm) {
    return;
  }
  auto pool = metal::new_scoped_memory_pool();
  auto heap_desc = MTL::HeapDescriptor::alloc()->init()->autorelease();
  heap_desc->setResourceOptions(resource_options);
  heap_desc->setSize(heap_size_);
  heap_ = NS::TransferPtr(device_->newHeap(heap_desc));
  residency_set_.insert(heap_.get());

  // Check if mmap mode is enabled via environment variable
  use_mmap_ = (getenv(\"MLX_USE_MMAP\") != nullptr);
  if (use_mmap_) {
    const char* mmap_dir = getenv(\"MLX_MMAP_DIR\");
    mmap_dir_ = mmap_dir ? mmap_dir : \"/tmp\";
    ::fprintf(stderr, \"[MLX SSD] mmap mode enabled, using directory: %s\\n\", mmap_dir_.c_str());
  }
}

MetalAllocator::~MetalAllocator() = default;

size_t MetalAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
};

size_t MetalAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, block_limit_);
  gc_limit_ = std::min(
      block_limit_,
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()));
  return limit;
};

size_t MetalAllocator::get_memory_limit() {
  return block_limit_;
}

size_t MetalAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  residency_set_.resize(wired_limit_);
  return limit;
}

Buffer MetalAllocator::malloc(size_t size) {
  // Metal doesn't like empty buffers
  if (size == 0) {
    return Buffer{nullptr};
  }

  // More helpful message if maximum buffer length is exceeded
  if (size > device_->maxBufferLength()) {
    std::ostringstream msg;
    msg << \"[metal::malloc] Attempting to allocate \" << size
        << \" bytes which is greater than\"
        << \" the maximum allowed buffer size of \" << device_->maxBufferLength()
        << \" bytes.\";
    throw std::runtime_error(msg.str());
  }

  // Align up memory
  if (size > vm_page_size) {
    size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  }

  // Try the cache
  std::unique_lock lk(mutex_);
  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    size_t mem_required = get_active_memory() + get_cache_memory() + size;

    // If we have a lot of memory pressure try to reclaim memory from the cache
    if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }

    // Allocate new buffer if needed
    if (num_resources_ >= resource_limit_) {
      std::ostringstream msg;
      msg << \"[metal::malloc] Resource limit (\" << resource_limit_
          << \") exceeded.\";
      throw std::runtime_error(msg.str());
    }
    lk.unlock();

    if (use_mmap_ && size > vm_page_size) {
      // Use mmap-backed allocation for large buffers (SSD offloading)
      buf = allocate_mmap_buffer(size);
    } else if (size < small_size_ && heap_) {
      buf = heap_->newBuffer(size, resource_options);
    } else {
      buf = device_->newBuffer(size, resource_options);
    }

    if (!buf) {
      std::ostringstream msg;
      msg << \"[malloc] Unable to allocate \" << size << \" bytes.\";
      throw std::runtime_error(msg.str());
    }
    lk.lock();
    num_resources_++;
    if (!buf->heap()) {
      residency_set_.insert(buf);
    }
  }

  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);

  // Maintain the cache below the requested limit
  if (get_cache_memory() > max_pool_size_) {
    num_resources_ -= buffer_cache_.release_cached_buffers(
        get_cache_memory() - max_pool_size_);
  }

  return Buffer{static_cast<void*>(buf)};
}

MTL::Buffer* MetalAllocator::allocate_mmap_buffer(size_t size) {
  // Generate a unique temp file name
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  std::string fname;
  do {
    fname = mmap_dir_ + \"/mlx_mmap_\";
    for (int i = 0; i < 16; i++) {
      char hex[2];
      snprintf(hex, sizeof(hex), \"%x\", dis(gen));
      fname += hex;
    }
  } while (access(fname.c_str(), F_OK) == 0);

  // Create and size the temp file
  int fd = open(fname.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    ::fprintf(stderr, \"[MLX SSD] Failed to create temp file %s: %d\\n\", fname.c_str(), errno);
    return nullptr;
  }

  // Set file size
  if (ftruncate(fd, size) < 0) {
    ::fprintf(stderr, \"[MLX SSD] Failed to ftruncate temp file: %d\\n\", errno);
    close(fd);
    unlink(fname.c_str());
    return nullptr;
  }

  // mmap the file
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int close_fd = fd; // save for close after unlink

  if (ptr == MAP_FAILED) {
    ::fprintf(stderr, \"[MLX SSD] mmap failed: %d\\n\", errno);
    close(close_fd);
    unlink(fname.c_str());
    return nullptr;
  }

  // Close the fd - OS keeps the mapping valid until unlink
  close(close_fd);

  // Immediately unlink the file so it doesn't persist in the filesystem
  // The mapping stays valid until we munmap
  unlink(fname.c_str());

  // Create Metal buffer backed by the mmap'd memory
  // newBufferWithBytesNoCopy: Metal directly reads/writes the mmap'd memory
  MTL::Buffer* metal_buf = device_->newBufferWithBytesNoCopy(
      ptr, size, MTL::ResourceOptionCPUCacheModeDefault, nullptr);

  if (!metal_buf) {
    ::fprintf(stderr, \"[MLX SSD] newBufferWithBytesNoCopy failed\\n\");
    munmap(ptr, size);
    return nullptr;
  }

  // Track this mmap allocation for proper cleanup in free()
  {
    std::unique_lock lk(mmap_mutex_);
    mmap_info_[metal_buf] = {ptr, size, fname};
  }

  ::fprintf(stderr, \"[MLX SSD] mmap buffer: %p, size: %zu\\n\", metal_buf, size);

  return metal_buf;
}

void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();

  // Check if this is an mmap-backed buffer
  bool is_mmap = false;
  void* mmap_ptr = nullptr;
  size_t mmap_size = 0;

  {
    std::unique_lock lk2(mmap_mutex_);
    auto it = mmap_info_.find(buf);
    if (it != mmap_info_.end()) {
      is_mmap = true;
      mmap_ptr = it->second.ptr;
      mmap_size = it->second.size;
      mmap_info_.erase(it);
    }
  }

  if (is_mmap) {
    // Unmap the SSD-backed memory
    if (mmap_ptr && mmap_size > 0) {
      munmap(mmap_ptr, mmap_size);
    }
  }

  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    num_resources_--;
    if (!buf->heap()) {
      residency_set_.erase(buf);
    }
    lk.unlock();
    auto pool = metal::new_scoped_memory_pool();
    buf->release();
  }
}

size_t MetalAllocator::size(Buffer buffer) const {
  return static_cast<MTL::Buffer*>(buffer.ptr())->length();
}

Buffer MetalAllocator::make_buffer(void* ptr, size_t size) {
  auto buf = device_->newBuffer(ptr, size, resource_options, nullptr);
  if (!buf) {
    return Buffer{nullptr};
  }
  std::unique_lock lk(mutex_);
  residency_set_.insert(buf);
  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);
  num_resources_++;
  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::release(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  num_resources_--;
  residency_set_.erase(buf);
  lk.unlock();
  auto pool = metal::new_scoped_memory_pool();
  buf->release();
}

MetalAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of MetalAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static MetalAllocator* allocator_ =
      new MetalAllocator(device(mlx::core::Device::gpu));
  return *allocator_;
}

} // namespace metal

size_t set_cache_limit(size_t limit) {
  return metal::allocator().set_cache_limit(limit);
}
size_t set_memory_limit(size_t limit) {
  return metal::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return metal::allocator().get_memory_limit();
}
size_t set_wired_limit(size_t limit) {
  if (limit > std::get<size_t>(
                  gpu::device_info(0).at(\"max_recommended_working_set_size\"))) {
    throw std::invalid_argument(
        \"[metal::set_wired_limit] Setting a wired limit larger than \"
        \"the maximum working set size is not allowed.\");
  }
  return metal::allocator().set_wired_limit(limit);
}
size_t get_active_memory() {
  return metal::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return metal::allocator().get_peak_memory();
}
void reset_peak_memory() {
  metal::allocator().reset_peak_memory();
}
size_t get_cache_memory() {
  return metal::allocator().get_cache_memory();
}
void clear_cache() {
  return metal::allocator().clear_cache();
}

} // namespace mlx::core
