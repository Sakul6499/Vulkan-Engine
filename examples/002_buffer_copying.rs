use vulkan_engine::{AbstractEngine, ComputeEngine};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
};

pub fn main() {
    env_logger::init();
    log::info!(
        "Logger initialized at max level set to {}",
        log::max_level()
    );
    log::info!("002 - Buffer Copying");

    // Start Compute Engine
    let compute_engine = ComputeEngine::new();

    // Print some information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Make Memory Allocator
    let memory_allocator =
        StandardMemoryAllocator::new_default(compute_engine.get_logical_device().get_device());

    // Source Buffer
    let source_content: Vec<i32> = (0..64).collect();
    let source_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        source_content,
    )
    .expect("failed to create source buffer");

    // Destination Buffer
    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        destination_content,
    )
    .expect("failed to create destination buffer");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            &engine.get_command_buffer_allocator(),
            engine.get_logical_device().get_queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                source_buffer.clone(),
                destination_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Assert results
    let source_content = source_buffer.read().unwrap();
    let destination_content = destination_buffer.read().unwrap();

    assert_eq!(&*source_content, &*destination_content);
    log::info!("Assertion passed");
}
