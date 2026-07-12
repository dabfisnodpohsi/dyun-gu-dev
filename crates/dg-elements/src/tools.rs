use std::sync::OnceLock;

use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, Packet, ParamField,
    ParamType, PortSchema, Result,
};

const INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
    required: true,
};
const HTTP_PUSH_FIELDS: &[&str] = &["url", "method"];
const HTTP_PUSH_PARAMS: &[ParamField] = &[
    ParamField {
        name: "url",
        ty: ParamType::Str,
        required: true,
    },
    ParamField {
        name: "method",
        ty: ParamType::Str,
        required: false,
    },
];

pub struct HttpPushRequest {
    pub url: String,
    pub method: String,
    pub packet: Packet,
}

pub trait HttpPushDriver: Send + Sync {
    fn post(&self, request: HttpPushRequest) -> Result<()>;
}

static HTTP_PUSH_DRIVER: OnceLock<Box<dyn HttpPushDriver>> = OnceLock::new();

pub fn install_http_push_driver(driver: Box<dyn HttpPushDriver>) -> Result<()> {
    HTTP_PUSH_DRIVER
        .set(driver)
        .map_err(|_| Error::Config("http_push driver already installed".to_string()))
}

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "http_push",
        input_ports: &[INPUT_PORT],
        output_ports: &[],
        params: HTTP_PUSH_PARAMS,
        validate: Some(validate_http_push),
        create: create_http_push,
    }
}

struct HttpPush {
    url: String,
    method: String,
}

impl Element for HttpPush {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        loop {
            let packet = match io.recv("in")? {
                Some(packet) => packet,
                None => {
                    if io.should_stop() {
                        return Err(Error::NotRunning);
                    }
                    continue;
                }
            };
            if packet.is_eos() {
                return Ok(());
            }
            let driver = HTTP_PUSH_DRIVER.get().ok_or_else(|| {
                Error::Runtime(format!(
                    "http_push node {} has no installed driver for {}",
                    io.name, self.url
                ))
            })?;
            driver
                .post(HttpPushRequest {
                    url: self.url.clone(),
                    method: self.method.clone(),
                    packet,
                })
                .map_err(|err| {
                    Error::Runtime(format!(
                        "http_push node {} failed posting to {}: {err}",
                        io.name, self.url
                    ))
                })?;
        }
    }
}

fn create_http_push(node: &NodeSpec) -> Result<CreatedElement> {
    let config = parse_http_push(node)?;
    Ok(CreatedElement {
        element: Box::new(HttpPush {
            url: config.url,
            method: config.method,
        }),
        handle: ElementHandle::None,
    })
}

fn validate_http_push(node: &NodeSpec) -> Result<()> {
    parse_http_push(node).map(|_| ())
}

struct HttpPushConfig {
    url: String,
    method: String,
}

fn parse_http_push(node: &NodeSpec) -> Result<HttpPushConfig> {
    let params = params_object(node)?;
    reject_unknown_fields(params, HTTP_PUSH_FIELDS)?;
    let url = match params.get("url") {
        Some(value) => value
            .as_str()
            .filter(|url| !url.is_empty())
            .ok_or_else(|| Error::Config("field url must be a non-empty string".to_string()))?,
        None => {
            return Err(Error::Config(
                "field url is required and must be a string".to_string(),
            ));
        }
    }
    .to_string();
    validate_http_url(&url)?;
    let method = params
        .get("method")
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| Error::Config("field method must be a string".to_string()))
        })
        .transpose()?
        .unwrap_or("POST");
    if method.is_empty() || method.chars().any(char::is_whitespace) {
        return Err(Error::Config(
            "field method must be a non-empty HTTP method".to_string(),
        ));
    }
    Ok(HttpPushConfig {
        url,
        method: method.to_ascii_uppercase(),
    })
}

fn validate_http_url(url: &str) -> Result<()> {
    let Some((scheme, rest)) = url.split_once("://") else {
        return Err(Error::Config(
            "field url must use the http:// or https:// scheme".to_string(),
        ));
    };
    if (!scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https"))
        || rest.is_empty()
        || rest.chars().any(char::is_whitespace)
    {
        return Err(Error::Config(
            "field url must use a non-empty http:// or https:// URL".to_string(),
        ));
    }
    Ok(())
}

fn params_object(node: &NodeSpec) -> Result<&serde_json::Map<String, serde_json::Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}

fn reject_unknown_fields(
    params: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
) -> Result<()> {
    for key in params.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Config(format!(
                "unknown field `{key}`; expected one of {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}
