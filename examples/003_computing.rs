use vulkan_engine::{AbstractEngine, ComputeEngine};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

mod shader {
    vulkano_shaders::shader! {ty: "compute", path: "shaders/003_computing.comp"}
}

pub fn main() {
    env_logger::init();
    log::info!(
        "Logger initialized at max level set to {}",
        log::max_level()
    );
    log::info!("003 - Computing");

    // Start Compute Engine
    let compute_engine = ComputeEngine::new();

    // Print information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Make Memory and DescriptorSet Allocator
    let memory_allocator =
        StandardMemoryAllocator::new_default(compute_engine.get_logical_device().get_device());
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(compute_engine.get_logical_device().get_device());

    // Prepare Data
    let data_iter = 0..65536;
    let data_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        data_iter,
    )
    .expect("failed to create source buffer");

    // Prepare Shader
    let shader = shader::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create shader module");

    // Prepare Compute Pipeline
    let compute_pipeline = ComputePipeline::new(
        compute_engine.get_logical_device().get_device(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    // Prepare Descriptor Set
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .expect("failed to create descriptor set");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            &engine.get_command_buffer_allocator(),
            engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch([1024, 1, 1])
            .unwrap();

        builder.build().unwrap()
    });

    // Assert results
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }
    log::info!("Assertion passed");
}
