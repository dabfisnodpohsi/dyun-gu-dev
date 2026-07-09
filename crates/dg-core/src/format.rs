/// Semantic tensor layout tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataFormat {
    Auto,
    N,
    NC,
    NCHW,
    NHWC,
    NC4HW,
    NC8HW,
    NCDHW,
    OIHW,
}
