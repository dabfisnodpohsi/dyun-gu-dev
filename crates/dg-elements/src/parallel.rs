use std::collections::BTreeMap;

use dg_graph::{
    CreatedElement, Element, ElementHandle, ElementIo, Error, NodeSpec, PortSchema, Result,
};

const MAX_BRANCHES: usize = 8;
const INPUT_PORT: PortSchema = PortSchema {
    name: "in",
    dtype: None,
};
const OUTPUT_PORTS: [PortSchema; MAX_BRANCHES] = [
    PortSchema {
        name: "out0",
        dtype: None,
    },
    PortSchema {
        name: "out1",
        dtype: None,
    },
    PortSchema {
        name: "out2",
        dtype: None,
    },
    PortSchema {
        name: "out3",
        dtype: None,
    },
    PortSchema {
        name: "out4",
        dtype: None,
    },
    PortSchema {
        name: "out5",
        dtype: None,
    },
    PortSchema {
        name: "out6",
        dtype: None,
    },
    PortSchema {
        name: "out7",
        dtype: None,
    },
];
const INPUT_PORTS: [PortSchema; MAX_BRANCHES] = [
    PortSchema {
        name: "in0",
        dtype: None,
    },
    PortSchema {
        name: "in1",
        dtype: None,
    },
    PortSchema {
        name: "in2",
        dtype: None,
    },
    PortSchema {
        name: "in3",
        dtype: None,
    },
    PortSchema {
        name: "in4",
        dtype: None,
    },
    PortSchema {
        name: "in5",
        dtype: None,
    },
    PortSchema {
        name: "in6",
        dtype: None,
    },
    PortSchema {
        name: "in7",
        dtype: None,
    },
];
const SINGLE_OUTPUT: [PortSchema; 1] = [PortSchema {
    name: "out",
    dtype: None,
}];

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "distributor",
        input_ports: &[INPUT_PORT],
        output_ports: &OUTPUT_PORTS,
        create: create_distributor,
    }
}

inventory::submit! {
    dg_graph::ElementDescriptor {
        kind: "converger",
        input_ports: &INPUT_PORTS,
        output_ports: &SINGLE_OUTPUT,
        create: create_converger,
    }
}

struct Distributor {
    strategy: DistributionStrategy,
}

#[derive(Clone, Copy)]
enum DistributionStrategy {
    RoundRobin,
    Broadcast,
}

struct Converger;

impl Element for Distributor {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        let output_ports = io
            .outputs
            .keys()
            .filter_map(|name| name.strip_prefix("out"))
            .filter_map(|index| index.parse::<usize>().ok())
            .collect::<Vec<_>>();
        if output_ports.is_empty() {
            return Err(Error::Config(
                "distributor requires at least one output".to_string(),
            ));
        }
        let mut next = 0usize;
        loop {
            let packet = match io.recv("in")? {
                Some(packet) => packet,
                None => continue,
            };
            if packet.is_eos() {
                io.broadcast_eos()?;
                return Ok(());
            }
            match self.strategy {
                DistributionStrategy::Broadcast => {
                    for index in &output_ports {
                        io.send(&format!("out{index}"), packet.clone())?;
                    }
                }
                DistributionStrategy::RoundRobin => {
                    let index = output_ports[next % output_ports.len()];
                    next = next.wrapping_add(1);
                    io.send(&format!("out{index}"), packet)?;
                }
            }
        }
    }
}

impl Element for Converger {
    fn run(self: Box<Self>, io: ElementIo) -> Result<()> {
        let mut finished = io
            .inputs
            .keys()
            .map(|name| (name.clone(), false))
            .collect::<BTreeMap<_, _>>();
        if finished.is_empty() {
            return Err(Error::Config(
                "converger requires at least one input".to_string(),
            ));
        }
        while finished.values().any(|done| !done) {
            let ports = finished
                .iter()
                .filter(|(_, done)| !**done)
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            let mut received = false;
            for port in ports {
                match io.recv(&port)? {
                    Some(packet) if packet.is_eos() => {
                        finished.insert(port, true);
                        received = true;
                    }
                    Some(packet) => {
                        io.send("out", packet)?;
                        received = true;
                    }
                    None => {}
                }
            }
            if !received && io.stop.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error::NotRunning);
            }
        }
        io.broadcast_eos()
    }
}

fn create_distributor(node: &NodeSpec) -> Result<CreatedElement> {
    let params = params_object(node)?;
    let strategy = match params
        .get("strategy")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("round_robin")
    {
        "round_robin" => DistributionStrategy::RoundRobin,
        "broadcast" => DistributionStrategy::Broadcast,
        value => {
            return Err(Error::Config(format!(
                "unknown distributor strategy {value}"
            )))
        }
    };
    Ok(CreatedElement {
        element: Box::new(Distributor { strategy }),
        handle: ElementHandle::None,
    })
}

fn create_converger(_node: &NodeSpec) -> Result<CreatedElement> {
    Ok(CreatedElement {
        element: Box::new(Converger),
        handle: ElementHandle::None,
    })
}

fn params_object(node: &NodeSpec) -> Result<&serde_json::Map<String, serde_json::Value>> {
    node.params
        .as_object()
        .ok_or_else(|| Error::Config(format!("node {} params must be an object", node.name)))
}
