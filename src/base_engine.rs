use std::sync::Arc;

use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer, instance::Instance,
};

use crate::LogicalDevice;

pub trait BaseEngine {
    /// Kills the engine as a safety measure in case other things like winit aren't killing the Engine part.
    fn kill(&self) {
        log::debug!("::: KILLING MAIN PROCESS :::");
        std::process::exit(std::process::id().try_into().unwrap());
    }

    /// Prints out information about the Vulkan API.
    fn print_api_information(instance: Arc<Instance>, log_level: log::Level) {
        log::log!(log_level, "API Version: {}", instance.api_version());
        log::log!(log_level, "Max API Version: {}", instance.max_api_version());
    }

    /// Computes a given operation on the GPU
    fn compute(&self, operation: &dyn (Fn(&Self) -> PrimaryAutoCommandBuffer));

    /// Returns the instance of the Vulkan API.
    fn get_instance(&self) -> Arc<Instance>;

    /// Returns the logical device.
    fn get_logical_device(&self) -> Arc<LogicalDevice>;
}
