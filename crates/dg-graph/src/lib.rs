#![forbid(unsafe_code)]

//! Graph execution, configuration, and pipeline orchestration.
//!
//! `dg-graph` owns the stream-oriented execution model for composing registered
//! elements into a runnable DAG. It also defines the versioned, format-agnostic
//! `GraphSpec` configuration model used to load and build graphs.

mod builtin;
mod element;
mod engine;
mod error;
mod packet;
mod registry;
mod spec;

pub use element::{CreatedElement, Element, ElementHandle, ElementIo, PortSchema, SinkCollector};
pub use engine::{watch, DataPipe, Graph, GraphDiff, GraphReport, WatchHandle};
pub use error::{Error, Result};
pub use packet::{Packet, PacketMeta, PacketPayload};
pub use registry::{
    create_element, element_ports, find_element, registered_elements, ElementDescriptor,
};
pub use spec::{ConnectionSpec, GraphFormat, GraphSpec, GraphSpecBuilder, NodeSpec, NodeTemplate};

// Bring built-in registrations into the inventory at link time.
// The module is intentionally private; the submit! calls are the important side effect.
