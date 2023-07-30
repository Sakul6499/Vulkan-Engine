use vulkan_engine::{AbstractEngine, ComputeEngine};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};

pub fn main() {
    env_logger::init();
    log::info!(
        "Logger initialized at max level set to {}",
        log::max_level()
    );
    log::info!("001 - Engine Init");

    let compute_engine = ComputeEngine::new();

    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    compute_engine.compute(&|engine: &ComputeEngine| {
        AutoCommandBufferBuilder::primary(
            &engine.get_command_buffer_allocator(),
            engine.get_logical_device().get_queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
        .build()
        .unwrap()
    });

    compute_engine.kill();
}
