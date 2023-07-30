use std::{sync::Arc, time::Instant};

use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    sync::GpuFuture,
    VulkanLibrary,
};

use crate::{AbstractEngine, LogicalDevice};

#[derive(Debug)]
pub struct ComputeEngine {
    instance: Arc<Instance>,
    logical_device: Arc<LogicalDevice>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl ComputeEngine {
    /// Creates a `ComputeEngine` instance and initializes everything needed for computing tasks.
    pub fn new() -> Self {
        log::debug!("ComputeEngine::startup");

        let instance = Self::create_instance();
        Self::print_api_information(instance.clone(), log::Level::Debug);

        let physical_device = Self::get_physical_device(instance.clone());
        let queue_family_index = Self::find_best_suited_queue_family(physical_device.clone());
        let logical_device = Self::create_logical_device(physical_device, queue_family_index);

        logical_device.print_interesting_information(log::Level::Debug);

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            logical_device.get_device(),
            StandardCommandBufferAllocatorCreateInfo {
                ..Default::default()
            },
        ));

        Self {
            instance,
            logical_device,
            command_buffer_allocator,
        }
    }

    /// Creates a Vulkan(o) instance.
    fn create_instance() -> Arc<Instance> {
        log::debug!("ComputeEngine::create_instance");

        let library = VulkanLibrary::new()
            .expect("failed to load Vulkan library. Make sure you have the Vulkan SDK installed.");

        Instance::new(library, InstanceCreateInfo::application_from_cargo_toml())
            .expect("failed to create Vulkan instance")
    }

    /// Queries all `PhysicalDevice`'s and returns the best match.
    fn get_physical_device(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
        log::debug!("ComputeEngine::get_physical_device");

        instance
            .enumerate_physical_devices()
            .expect("failed enumerating physical devices")
            .filter(|physical_device: &Arc<PhysicalDevice>| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .any(|queue_family| queue_family.queue_flags.contains(QueueFlags::COMPUTE))
            })
            .min_by_key(|physical_device: &Arc<PhysicalDevice>| {
                match physical_device.properties().device_type {
                    vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 0,
                    vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 1,
                    vulkano::device::physical::PhysicalDeviceType::VirtualGpu => 2,
                    vulkano::device::physical::PhysicalDeviceType::Cpu => 3,
                    vulkano::device::physical::PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no physical device found or available")
    }

    /// Finds the best suited `QueueFamily` of a given device or none if no suitable family was found.
    fn find_best_suited_queue_family(physical_device: Arc<PhysicalDevice>) -> u32 {
        log::debug!("ComputeEngine::find_best_suited_queue_family");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE)) // Find a compute capable queue family
            .expect("no suitable queue family found") as u32;

        log::debug!("Queue family index: {}", queue_family_index);
        queue_family_index
    }

    /// Creates a `LogicalDevice` given a `PhysicalDevice` and a `QueueFamilyIndex`.
    fn create_logical_device(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
    ) -> Arc<LogicalDevice> {
        log::debug!("ComputeEngine::create_logical_device");

        let (device, raw_queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create logical device");

        let queues: Vec<Arc<Queue>> = raw_queues.collect();

        let logical_device = LogicalDevice::new(device, queue_family_index, queues);
        Arc::new(logical_device)
    }
}

impl AbstractEngine<StandardCommandBufferAllocator> for ComputeEngine {
    fn compute(&self, operation: &dyn (Fn(&Self) -> PrimaryAutoCommandBuffer)) {
        let command_buffer = operation(self);

        #[cfg(debug_assertions)]
        let start_fence = Instant::now();

        command_buffer
            .execute(self.get_logical_device().get_first_queue())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        #[cfg(debug_assertions)]
        {
            let end_fence = Instant::now();
            log::debug!(
                "Compute operation took: {}ms",
                end_fence.duration_since(start_fence).as_millis()
            );
        }
    }

    fn get_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    fn get_logical_device(&self) -> Arc<LogicalDevice> {
        self.logical_device.clone()
    }

    fn get_command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }
}

impl Default for ComputeEngine {
    fn default() -> Self {
        Self::new()
    }
}
