// use vulkan_engine::{BaseEngine, ComputeEngine};
// use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};

// pub fn main() {
//     let compute_engine = ComputeEngine::new();

//     ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

//     compute_engine.compute(&|engine: &ComputeEngine| {
//         AutoCommandBufferBuilder::primary(
//             engine.get_logical_device().get_device(),
//             engine.get_logical_device().get_queue_family_index(),
//             CommandBufferUsage::OneTimeSubmit,
//         )
//         .unwrap()
//         .build()
//         .unwrap()
//     });

//     compute_engine.kill();
// }
