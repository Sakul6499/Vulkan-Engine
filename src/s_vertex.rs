use bytemuck::Zeroable;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Vertex, BufferContents, Zeroable, Copy, Clone, PartialEq, PartialOrd)]
#[repr(C)]
pub struct SVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}
