/// An axis-aligned bounding box in image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn area(self) -> f32 {
        self.w.max(0.0) * self.h.max(0.0)
    }
}

/// A detected object and its confidence metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
    pub class_id: u32,
    pub label: Option<String>,
}

impl Detection {
    pub fn new(bbox: BBox, score: f32, class_id: u32) -> Self {
        Self {
            bbox,
            score,
            class_id,
            label: None,
        }
    }
}
