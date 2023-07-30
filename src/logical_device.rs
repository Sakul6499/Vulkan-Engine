use std::sync::Arc;

use vulkano::device::{physical::PhysicalDevice, Device, Properties, Queue};

/// A LogicalDevice is a wrapper around a Vulkano Device and it's queues.
#[derive(Clone, Debug)]
pub struct LogicalDevice {
    /// The Vulkano Device
    device: Arc<Device>,
    /// The QueueFamilyIndex
    queue_family_index: u32,
    /// A Vector of all Queues in this family
    queues: Vec<Arc<Queue>>,
}

impl LogicalDevice {
    /// Creates a new LogicalDevice from a given Device and QueueFamilyIndex.
    pub fn new(device: Arc<Device>, queue_family_index: u32, queues: Vec<Arc<Queue>>) -> Self {
        Self {
            device,
            queue_family_index,
            queues,
        }
    }

    /// Returns the device of the LogicalDevice.
    pub fn get_device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Returns the queue family index of the LogicalDevice.
    pub fn get_queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the queues of the LogicalDevice.
    pub fn get_queues(&self) -> Vec<Arc<Queue>> {
        self.queues.clone()
    }

    /// Returns the first queue of the LogicalDevice.
    pub fn get_first_queue(&self) -> Arc<Queue> {
        self.queues[0].clone()
    }

    /// Prints out all enabled extensions of the LogicalDevice.
    pub fn print_enabled_extensions(&self, log_level: log::Level) {
        let extensions = self.device.enabled_extensions();
        log::log!(log_level, "Enabled Extensions: {:#?}", extensions);
    }

    /// Prints out all enabled features of the LogicalDevice.
    pub fn print_enabled_features(&self, log_level: log::Level) {
        let features = self.device.enabled_features();
        log::log!(log_level, "Enabled Features: {:#?}", features);
    }

    pub fn print_interesting_information(&self, log_level: log::Level) {
        let properties = self.device.physical_device().properties();
        log::log!(log_level, "Device Name: {}", self.get_device_name());
        log::log!(log_level, "Device Type: {:?}", properties.device_type);
        log::log!(log_level, "Driver Version: {}", properties.driver_version);
        log::log!(log_level, "Vendor ID: {}", properties.vendor_id);
        log::log!(log_level, "Device ID: {}", properties.device_id);
        log::log!(log_level, "API Version: {}", properties.api_version);
        log::log!(log_level, "Driver ID: {:?}", properties.driver_id);
        log::log!(log_level, "Device UUID: {:?}", properties.device_uuid);
        log::log!(
            log_level,
            "Pipeline Cache UUID: {:?}",
            properties.pipeline_cache_uuid
        );
        log::log!(
            log_level,
            "Device Max Memory Allocation Count: {}",
            self.get_device_max_memory_allocation_count()
        );
        log::log!(
            log_level,
            "Device Max Memory Allocation Size: {:?}",
            self.get_device_max_memory_allocation_size()
        );
        log::log!(
            log_level,
            "Device Max Compute Shared Memory Size: {}",
            self.get_device_max_compute_shared_memory_size()
        );
        log::log!(
            log_level,
            "Device Max Buffer Size: {:?}",
            self.get_device_max_buffer_size()
        );

        self.print_enabled_extensions(log_level);
        self.print_enabled_features(log_level);
    }

    pub fn get_physical_device(&self) -> Arc<PhysicalDevice> {
        self.device.physical_device().clone()
    }

    pub fn get_physical_properties(&self) -> Properties {
        self.get_physical_device().properties().clone()
    }

    pub fn get_device_name(&self) -> String {
        self.get_physical_properties().device_name
    }

    pub fn get_device_max_memory_allocation_count(&self) -> u32 {
        self.get_physical_properties().max_memory_allocation_count
    }

    pub fn get_device_max_memory_allocation_size(&self) -> Option<u64> {
        self.get_physical_properties().max_memory_allocation_size
    }

    pub fn get_device_max_compute_shared_memory_size(&self) -> u32 {
        self.get_physical_properties()
            .max_compute_shared_memory_size
    }

    pub fn get_device_max_buffer_size(&self) -> Option<u64> {
        self.get_physical_properties().max_buffer_size
    }
}

impl std::fmt::Display for LogicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Device Name: {}; Queue Family Index: {}; Queue Count: {}",
            self.get_device_name(),
            self.queue_family_index,
            self.queues.len()
        )
    }
}
