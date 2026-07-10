use crate::element::{CreatedElement, PortSchema};
use crate::error::{Error, Result};
use crate::schema::ParamField;
use crate::spec::NodeSpec;

pub struct ElementDescriptor {
    pub kind: &'static str,
    pub input_ports: &'static [PortSchema],
    pub output_ports: &'static [PortSchema],
    pub params: &'static [ParamField],
    pub validate: Option<fn(&NodeSpec) -> Result<()>>,
    pub create: fn(&NodeSpec) -> Result<CreatedElement>,
}

inventory::collect!(ElementDescriptor);

pub fn registered_elements() -> Vec<&'static ElementDescriptor> {
    inventory::iter::<ElementDescriptor>.into_iter().collect()
}

pub fn find_element(kind: &str) -> Option<&'static ElementDescriptor> {
    registered_elements()
        .into_iter()
        .find(|descriptor| descriptor.kind == kind)
}

pub fn create_element(node: &NodeSpec) -> Result<CreatedElement> {
    let descriptor =
        find_element(&node.kind).ok_or_else(|| Error::UnknownNodeKind(node.kind.clone()))?;
    (descriptor.create)(node)
}

pub fn validate_element(node: &NodeSpec) -> Result<()> {
    let descriptor =
        find_element(&node.kind).ok_or_else(|| Error::UnknownNodeKind(node.kind.clone()))?;
    if let Some(validate) = descriptor.validate {
        validate(node)?;
    }
    Ok(())
}

pub fn element_ports(kind: &str) -> Result<(&'static [PortSchema], &'static [PortSchema])> {
    let descriptor = find_element(kind).ok_or_else(|| Error::UnknownNodeKind(kind.to_string()))?;
    Ok((descriptor.input_ports, descriptor.output_ports))
}
