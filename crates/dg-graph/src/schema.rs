use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::registry::find_element;

#[derive(Clone, Copy, Debug)]
pub enum ParamType {
    Bool,
    Uint,
    Int,
    Float,
    Str,
    Enum(&'static [&'static str]),
    Array(&'static ParamType),
    Object,
}

#[derive(Clone, Copy, Debug)]
pub struct ParamField {
    pub name: &'static str,
    pub ty: ParamType,
    pub required: bool,
}

pub fn params_json_schema(params: &[ParamField]) -> Value {
    let mut schema = Map::new();
    schema.insert(
        "$schema".to_string(),
        Value::String("https://json-schema.org/draft/2020-12/schema".to_string()),
    );
    schema.insert("type".to_string(), Value::String("object".to_string()));
    schema.insert("additionalProperties".to_string(), Value::Bool(false));

    let mut properties = Map::new();
    let mut required = Vec::new();
    for field in params {
        properties.insert(field.name.to_string(), param_type_schema(field.ty));
        if field.required {
            required.push(Value::String(field.name.to_string()));
        }
    }
    schema.insert("properties".to_string(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".to_string(), Value::Array(required));
    }
    Value::Object(schema)
}

pub fn element_params_schema(kind: &str) -> Option<Value> {
    find_element(kind).map(|descriptor| params_json_schema(descriptor.params))
}

pub fn all_element_schemas() -> BTreeMap<&'static str, Value> {
    find_registered_elements()
        .map(|descriptor| (descriptor.kind, params_json_schema(descriptor.params)))
        .collect()
}

fn find_registered_elements() -> impl Iterator<Item = &'static crate::registry::ElementDescriptor> {
    crate::registry::registered_elements().into_iter()
}

fn param_type_schema(param_type: ParamType) -> Value {
    match param_type {
        ParamType::Bool => type_value("boolean"),
        ParamType::Uint => {
            let mut schema = type_object("integer");
            schema.insert("minimum".to_string(), Value::from(0));
            Value::Object(schema)
        }
        ParamType::Int => type_value("integer"),
        ParamType::Float => type_value("number"),
        ParamType::Str => type_value("string"),
        ParamType::Enum(values) => {
            let mut schema = type_object("string");
            schema.insert(
                "enum".to_string(),
                Value::Array(
                    values
                        .iter()
                        .map(|value| Value::String((*value).to_string()))
                        .collect(),
                ),
            );
            Value::Object(schema)
        }
        ParamType::Array(inner) => {
            let mut schema = type_object("array");
            schema.insert("items".to_string(), param_type_schema(*inner));
            Value::Object(schema)
        }
        ParamType::Object => type_value("object"),
    }
}

fn type_value(type_name: &str) -> Value {
    Value::Object(type_object(type_name))
}

fn type_object(type_name: &str) -> Map<String, Value> {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String(type_name.to_string()));
    schema
}

#[cfg(test)]
mod tests {
    use super::element_params_schema;

    #[test]
    fn every_registered_element_has_a_parameter_schema() {
        for descriptor in crate::registry::registered_elements() {
            let schema = element_params_schema(descriptor.kind)
                .expect("registered elements must have parameter schemas");
            assert_eq!(schema["type"], "object");
            assert_eq!(schema["additionalProperties"], false);
            let properties = schema["properties"]
                .as_object()
                .expect("schema properties must be an object");
            if let Some(required) = schema.get("required") {
                for name in required.as_array().expect("required must be an array") {
                    let name = name.as_str().expect("required names must be strings");
                    assert!(
                        properties.contains_key(name),
                        "missing required field {name}"
                    );
                }
            }
            for property in properties.values() {
                if let Some(values) = property.get("enum") {
                    assert!(
                        !values.as_array().expect("enum must be an array").is_empty(),
                        "enum fields must have at least one value"
                    );
                }
            }
        }
    }
}
