use std::time::Instant;

use image::{ImageBuffer, Rgba};
use vulkan_engine::{AbstractEngine, ComputeEngine, SVertex};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassContents,
    },
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    single_pass_renderpass,
};

mod shader_vertex {
    vulkano_shaders::shader! {ty: "vertex", path: "shaders/004_graphical_pipeline.vert"}
}

mod shader_fragment {
    vulkano_shaders::shader! {ty: "fragment", path: "shaders/004_graphical_pipeline.frag"}
}

pub fn main() {
    env_logger::init();
    log::info!(
        "Logger initialized at max level set to {}",
        log::max_level()
    );
    log::info!("005 - Graphical Pipeline");

    // Prepare Engine
    let compute_engine = ComputeEngine::new();

    // Print information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Set vertices for triangle
    let vertex1 = SVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = SVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = SVertex {
        position: [0.5, -0.25],
    };

    // Make Memory and CommandBuffer Allocator
    let memory_allocator =
        StandardMemoryAllocator::new_default(compute_engine.get_logical_device().get_device());
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        compute_engine.get_logical_device().get_device(),
        StandardCommandBufferAllocatorCreateInfo {
            ..Default::default()
        },
    );

    // Create vertex buffer
    let vertex_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();

    // Create Output buffer
    let output_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .unwrap();

    // Load Shaders
    let vertex_shader = shader_vertex::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create vertex shader module");
    let fragment_shader = shader_fragment::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create fragment shader module");

    // Define Viewport
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    // Create RenderPass (prepare "rendering mode")
    // Defines the format and way of the image to be rendered
    let render_pass = single_pass_renderpass!(
        compute_engine.get_logical_device().get_device(),
        attachments: {
            color: {
                load: Clear,    // Tells the GPU to clear the image when entering RenderPass
                store: Store,   // Tells the GPU to store any outputs to our image
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    // Create Image
    let image = StorageImage::new(
        &memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(compute_engine.get_logical_device().get_queue_family_index()),
    )
    .unwrap();

    // Create ImageView
    // Needed as a link between the CPU and the GPU
    let view = ImageView::new_default(image.clone()).unwrap();

    // Create FrameBuffer
    // Used to store images that are rendered.
    // But also handles attachments.
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    // Create GraphicsPipeline
    let pipeline = GraphicsPipeline::start()
        // Defines the layout of our Vertex object
        .vertex_input_state(SVertex::per_vertex())
        // Defines the entry point of our vertex shader
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        // Defines the primitive type (e.g. triangles, quads, etc.)
        // Default is triangles.
        .input_assembly_state(InputAssemblyState::new())
        // Defines the viewport
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        // Defines the entry point of our fragment shader
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        // Defines the render pass
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        // Build it! :)
        .build(compute_engine.get_logical_device().get_device())
        .unwrap();

    // Submit Command Buffer for Computation
    compute_engine.compute(&|compute_engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            compute_engine.get_logical_device().get_queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(
                3, // Vertex count
                1, // Instance count
                0, // First vertex
                0, // First instance
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                output_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Save results
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let buffer_content = output_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("004_graphical_pipeline.png").unwrap();

    #[cfg(debug_assertions)]
    let end = Instant::now();

    log::info!("Successfully saved image");

    #[cfg(debug_assertions)]
    log::debug!(
        "Storing image took: {}ms",
        end.duration_since(start).as_millis()
    );
}
