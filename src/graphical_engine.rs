use std::sync::Arc;
use std::time::Instant;

use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, ImageUsage, SwapchainImage},
    instance::InstanceExtensions,
    instance::{Instance, InstanceCreateInfo},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
    sync::GpuFuture,
    VulkanLibrary,
};

use crate::{AbstractEngine, LogicalDevice};

pub struct GraphicalEngine {
    instance: Arc<Instance>,
    logical_device: Arc<LogicalDevice>,
    window: Arc<Surface>,
    swap_chain: Arc<Swapchain>,
    swap_chain_images: Vec<Arc<SwapchainImage>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl GraphicalEngine {
    /// Creates a new instance of the `GraphicalEngine`.
    pub fn make_instance() -> Arc<Instance> {
        log::debug!("GraphicalEngine::make_instance");

        let instance = Self::create_instance();
        Self::print_api_information(instance.clone(), log::Level::Debug);

        instance
    }

    /// Creates a `GraphicalEngine` instance and initializes everything needed for graphical tasks.
    pub fn new(instance: Arc<Instance>, window: Arc<Surface>) -> Self {
        log::debug!("GraphicalEngine::startup");

        let (physical_device, queue_family_index) =
            Self::get_physical_device(instance.clone(), window.clone());

        let logical_device = Self::create_logical_device(physical_device, queue_family_index);

        logical_device.print_interesting_information(log::Level::Debug);

        let (swap_chain, swap_chain_images) =
            Self::create_swap_chain(logical_device.clone(), window.clone());

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            logical_device.get_device(),
            StandardCommandBufferAllocatorCreateInfo {
                ..Default::default()
            },
        ));

        Self {
            instance,
            logical_device,
            window,
            swap_chain,
            swap_chain_images,
            command_buffer_allocator,
        }
    }

    /// Retrieves the required extensions to run the engine.
    fn retrieve_required_instance_extensions(library: &VulkanLibrary) -> InstanceExtensions {
        log::debug!("GraphicalEngine::retrieve_required_instance_extensions");

        vulkano_win::required_extensions(library)
    }

    /// Returns the required device extensions.
    fn retrieve_required_device_extensions() -> DeviceExtensions {
        log::debug!("GraphicalEngine::retrieve_required_device_extensions");

        DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        }
    }

    /// Creates a Vulkan(o) instance.
    fn create_instance() -> Arc<Instance> {
        log::debug!("GraphicalEngine::create_instance");

        let library = VulkanLibrary::new()
            .expect("failed to load Vulkan library. Make sure you have the Vulkan SDK installed.");

        Instance::new(
            library.clone(),
            InstanceCreateInfo {
                enabled_extensions: GraphicalEngine::retrieve_required_instance_extensions(
                    &library,
                ),
                ..InstanceCreateInfo::application_from_cargo_toml()
            },
        )
        .expect("failed to create Vulkan instance")
    }

    /// Queries all `PhysicalDevice`'s and returns the best match.
    /// Returns a tuple of the best `PhysicalDevice` with the best `QueueFamily` index.
    fn get_physical_device(
        instance: Arc<Instance>,
        window: Arc<Surface>,
    ) -> (Arc<PhysicalDevice>, u32) {
        log::debug!("GraphicalEngine::get_physical_device");

        instance
            .enumerate_physical_devices()
            .expect("failed enumerating physical devices")
            // Filter out any devices that don't have a queue family that supports graphics
            .filter(|physical_device: &Arc<PhysicalDevice>| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .any(|queue_family| queue_family.queue_flags.contains(QueueFlags::GRAPHICS))
            })
            // Filter out any device that doesn't support ur required device extensions
            .filter(|physical_device: &Arc<PhysicalDevice>| {
                physical_device
                    .supported_extensions()
                    .contains(&GraphicalEngine::retrieve_required_device_extensions())
            })
            .map(|physical_device| {
                let best_queue_family_index =
                    GraphicalEngine::find_best_suited_queue_family(physical_device.clone());

                (physical_device, best_queue_family_index)
            })
            // Filter out any device that doesn't support our VkSurface
            .filter(|(physical_device, queue_family_index)| {
                physical_device
                    .surface_support(*queue_family_index, &window)
                    .unwrap_or(false)
            })
            .min_by_key(|(physical_device, _queue_family_index)| {
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
        log::debug!("GraphicalEngine::find_best_suited_queue_family");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS)) // Find a compute capable queue family
            .expect("no suitable queue family found") as u32;

        log::debug!("Queue family index: {}", queue_family_index);
        queue_family_index
    }

    /// Creates a `LogicalDevice` given a `PhysicalDevice` and a `QueueFamilyIndex`.
    fn create_logical_device(
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
    ) -> Arc<LogicalDevice> {
        log::debug!("GraphicalEngine::create_logical_device");

        let (device, raw_queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: GraphicalEngine::retrieve_required_device_extensions(),
                ..Default::default()
            },
        )
        .expect("failed to create logical device");

        let queues: Vec<Arc<Queue>> = raw_queues.collect();

        let logical_device = LogicalDevice::new(device, queue_family_index, queues);
        Arc::new(logical_device)
    }

    /// Creates a `Swapchain` given a `LogicalDevice` and a `Window`.
    /// Vulkan uses `Swapchain`s to store images while they are still ready and swaps them out once a new image is ready to be displayed (i.e. finished rendering).
    fn create_swap_chain(
        logical_device: Arc<LogicalDevice>,
        window: Arc<Surface>,
    ) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
        // Get surface capabilities
        let capabilities = logical_device
            .get_physical_device()
            .surface_capabilities(&window, Default::default())
            .expect("failed to get surface capabilities");

        // Get alphas
        let composite_alpha = capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .expect("no composite alpha found!");

        // Get image format
        let image_format = Some(
            logical_device
                .get_physical_device()
                .surface_formats(&window, Default::default())
                .unwrap()[0]
                .0,
        );

        // Create Swap Chain
        Swapchain::new(
            logical_device.get_device(),
            window,
            SwapchainCreateInfo {
                // How many buffers (images) are in the swap chain
                min_image_count: capabilities.min_image_count + 1,
                // Format of the images
                image_format,
                // Dimensions of the images
                image_extent: [0, 0],
                // Usage of the images
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                // Alpha of the images
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// Recreates the `SwapChain` and `SwapChainImages`, while also rebuilding the `Framebuffer`s given a `RenderPass` is submitted.
    ///
    /// Can return `None` on `SwapchainCreationError::ImageExtentNotSupported` which **should be ignored**.
    pub fn recreate_swap_chain_and_images(
        &mut self,
        render_pass: Arc<RenderPass>,
    ) -> Option<Vec<Arc<Framebuffer>>> {
        log::debug!("GraphicalEngine::recreate_swap_chain");

        let (new_swapchain, new_images) =
            match self.get_swap_chain().recreate(SwapchainCreateInfo {
                image_extent: [0, 0],
                ..self.get_swap_chain().create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return None,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

        self.swap_chain = new_swapchain;
        self.swap_chain_images = new_images;
        Some(self.create_frame_buffers(render_pass))
    }

    /// Creates a `RenderPass`.
    /// A `RenderPass` is a collection of `Attachment`s and `Subpass`es.
    /// It defines how an image on the `Swapchain` is being used and how it is being rendered.
    pub fn create_render_pass(&self) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            self.logical_device.get_device(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: self.swap_chain.image_format(), // Must be same as swap chain
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap()
    }

    /// Creates a `Framebuffer` from a given `RenderPass` and `SwapChainImages`.
    /// A `Framebuffer` wraps around `SwapchainImage`'s and creates `ImageView`s from them given the correct format from the given `RenderPass`.
    pub fn create_frame_buffers(&self, render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
        let buffers = self
            .swap_chain_images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Framebuffer>>>();

        buffers
    }

    /// Returns the `EngineWindow`
    pub fn get_window(&self) -> Arc<Surface> {
        self.window.clone()
    }

    /// Returns the `Swapchain`
    pub fn get_swap_chain(&self) -> Arc<Swapchain> {
        self.swap_chain.clone()
    }

    /// Returns the `SwapchainImage`s
    pub fn get_swap_chain_images(&self) -> Vec<Arc<SwapchainImage>> {
        self.swap_chain_images.clone()
    }
}

impl AbstractEngine<StandardCommandBufferAllocator> for GraphicalEngine {
    /// Runs compute operations on the `GraphicalEngine`.
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

    /// Returns the `Instance` Arc.
    fn get_instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    /// Returns the `LogicalDevice` Arc.
    fn get_logical_device(&self) -> Arc<LogicalDevice> {
        self.logical_device.clone()
    }

    fn get_command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }
}
