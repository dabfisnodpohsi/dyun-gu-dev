#![forbid(unsafe_code)]

//! Device/core scheduling and load balancing.
//!
//! `dg-scheduler` owns the pure Rust resource planner for device selection,
//! core masks, affinity, and least-loaded allocation. It intentionally stays
//! independent from graph/runtime wiring in this milestone; later crates will
//! consume the scheduler to map backend requests onto concrete device/core
//! placements.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use dg_core::{DeployMode, DeviceKind};
use thiserror::Error;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("topology cannot be empty")]
    EmptyTopology,
    #[error("duplicate device: kind={kind:?}, id={id}")]
    DuplicateDevice { kind: DeviceKind, id: u16 },
    #[error("device {kind:?}:{id} has no cores")]
    EmptyDevice { kind: DeviceKind, id: u16 },
    #[error("core id {core_id} on device {kind:?}:{id} exceeds mask capacity")]
    CoreIdOutOfRange {
        kind: DeviceKind,
        id: u16,
        core_id: u8,
    },
    #[error("duplicate core id {core_id} on device {kind:?}:{id}")]
    DuplicateCore {
        kind: DeviceKind,
        id: u16,
        core_id: u8,
    },
    #[error("unknown device {kind:?}:{id}")]
    UnknownDevice { kind: DeviceKind, id: u16 },
    #[error("no device of kind {kind:?} matched the request")]
    NoMatchingDevice { kind: DeviceKind },
    #[error("requested core mask {mask:#010x} does not match any core")]
    InvalidCoreMask { mask: u32 },
    #[error("requested core mask {mask:#010x} selects unavailable core {core_id}")]
    MissingCore { mask: u32, core_id: u8 },
    #[error("scheduler has no available cores for the request")]
    NoAvailableCore,
}

/// A schedulable device with a numeric card id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub kind: DeviceKind,
    pub id: u16,
    pub cores: Vec<Core>,
}

/// A schedulable core identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Core {
    pub id: u8,
}

/// A device/core topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    deployment: DeployMode,
    devices: Vec<Device>,
}

impl Topology {
    pub fn new(deployment: DeployMode, devices: Vec<Device>) -> Result<Self> {
        validate_topology(&devices)?;
        Ok(Self {
            deployment,
            devices,
        })
    }

    pub fn single_chip(kind: DeviceKind, core_count: u8) -> Result<Self> {
        Self::new(
            DeployMode::SoC,
            vec![Device {
                kind,
                id: 0,
                cores: cores_from_count(core_count),
            }],
        )
    }

    pub fn single_card_multi_core(kind: DeviceKind, card: u16, core_count: u8) -> Result<Self> {
        Self::new(
            DeployMode::Host,
            vec![Device {
                kind,
                id: card,
                cores: cores_from_count(core_count),
            }],
        )
    }

    pub fn multi_card_multi_core<I>(kind: DeviceKind, cards: I) -> Result<Self>
    where
        I: IntoIterator<Item = (u16, u8)>,
    {
        let devices = cards
            .into_iter()
            .map(|(id, core_count)| Device {
                kind,
                id,
                cores: cores_from_count(core_count),
            })
            .collect();
        Self::new(DeployMode::Host, devices)
    }

    pub fn deployment(&self) -> DeployMode {
        self.deployment
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }
}

/// Core selection expressed as a bitmask-compatible request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreSelection {
    Auto,
    Single(u8),
    Mask(u32),
    All,
}

impl CoreSelection {
    fn contains(self, core_id: u8) -> bool {
        match self {
            Self::Auto | Self::All => true,
            Self::Single(selected) => selected == core_id,
            Self::Mask(mask) => mask & (1u32 << u32::from(core_id)) != 0,
        }
    }

    fn is_explicit(self) -> bool {
        !matches!(self, Self::Auto)
    }
}

/// Scheduling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingMode {
    Auto,
    Explicit,
}

/// Request used to acquire a device/core lease.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub kind: DeviceKind,
    pub device_id: Option<u16>,
    pub mode: SchedulingMode,
    pub core_selection: CoreSelection,
    pub affinity_key: Option<String>,
}

impl Request {
    pub fn auto(kind: DeviceKind) -> Self {
        Self {
            kind,
            device_id: None,
            mode: SchedulingMode::Auto,
            core_selection: CoreSelection::Auto,
            affinity_key: None,
        }
    }

    pub fn explicit(kind: DeviceKind, device_id: u16, core_selection: CoreSelection) -> Self {
        Self {
            kind,
            device_id: Some(device_id),
            mode: SchedulingMode::Explicit,
            core_selection,
            affinity_key: None,
        }
    }

    pub fn with_affinity_key(mut self, affinity_key: impl Into<String>) -> Self {
        self.affinity_key = Some(affinity_key.into());
        self
    }

    pub fn with_device_id(mut self, device_id: u16) -> Self {
        self.device_id = Some(device_id);
        self
    }

    pub fn with_core_selection(mut self, core_selection: CoreSelection) -> Self {
        self.core_selection = core_selection;
        self
    }
}

/// Snapshot of a core including current load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreLoad {
    pub id: u8,
    pub load: usize,
}

/// Snapshot of a device including current core loads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLoad {
    pub kind: DeviceKind,
    pub id: u16,
    pub cores: Vec<CoreLoad>,
}

#[derive(Clone, Debug)]
struct CoreState {
    id: u8,
    load: usize,
}

#[derive(Clone, Debug)]
struct DeviceState {
    kind: DeviceKind,
    id: u16,
    cores: Vec<CoreState>,
}

#[derive(Clone, Debug)]
struct SchedulerState {
    devices: Vec<DeviceState>,
    affinity: HashMap<String, Allocation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Allocation {
    device_index: usize,
    core_index: usize,
}

/// Scheduler that performs least-loaded allocation with optional affinity.
#[derive(Clone, Debug)]
pub struct Scheduler {
    topology: Topology,
    state: Arc<Mutex<SchedulerState>>,
}

impl Scheduler {
    pub fn new(topology: Topology) -> Result<Self> {
        let state = SchedulerState::from_topology(&topology)?;
        Ok(Self {
            topology,
            state: Arc::new(Mutex::new(state)),
        })
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn snapshot(&self) -> Result<Vec<DeviceLoad>> {
        let state = self.state.lock().map_err(|_| Error::NoAvailableCore)?;
        Ok(state.snapshot())
    }

    pub fn acquire(&self, request: Request) -> Result<Lease> {
        let mut state = self.state.lock().map_err(|_| Error::NoAvailableCore)?;
        let allocation = state.acquire(request)?;
        Ok(Lease {
            state: Arc::clone(&self.state),
            allocation,
        })
    }
}

/// RAII lease returned by the scheduler.
#[derive(Debug)]
pub struct Lease {
    state: Arc<Mutex<SchedulerState>>,
    allocation: Allocation,
}

impl Lease {
    pub fn device(&self) -> (DeviceKind, u16) {
        let state = self.state.lock().expect("scheduler state poisoned");
        let device = &state.devices[self.allocation.device_index];
        (device.kind, device.id)
    }

    pub fn core_id(&self) -> u8 {
        let state = self.state.lock().expect("scheduler state poisoned");
        state.devices[self.allocation.device_index].cores[self.allocation.core_index].id
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.release(self.allocation);
        }
    }
}

impl SchedulerState {
    fn from_topology(topology: &Topology) -> Result<Self> {
        validate_topology(&topology.devices)?;
        let devices = topology
            .devices
            .iter()
            .map(|device| DeviceState {
                kind: device.kind,
                id: device.id,
                cores: device
                    .cores
                    .iter()
                    .map(|core| CoreState {
                        id: core.id,
                        load: 0,
                    })
                    .collect(),
            })
            .collect();
        Ok(Self {
            devices,
            affinity: HashMap::new(),
        })
    }

    fn snapshot(&self) -> Vec<DeviceLoad> {
        self.devices
            .iter()
            .map(|device| DeviceLoad {
                kind: device.kind,
                id: device.id,
                cores: device
                    .cores
                    .iter()
                    .map(|core| CoreLoad {
                        id: core.id,
                        load: core.load,
                    })
                    .collect(),
            })
            .collect()
    }

    fn acquire(&mut self, request: Request) -> Result<Allocation> {
        let device_indexes = self
            .devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.kind == request.kind)
            .filter(|(_, device)| request.device_id.is_none_or(|id| device.id == id))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        if device_indexes.is_empty() {
            return if request.device_id.is_some() {
                Err(Error::UnknownDevice {
                    kind: request.kind,
                    id: request.device_id.expect("checked is_some"),
                })
            } else {
                Err(Error::NoMatchingDevice { kind: request.kind })
            };
        }

        if request.mode == SchedulingMode::Auto {
            if let Some(key) = &request.affinity_key {
                if let Some(allocation) = self.affinity.get(key).copied() {
                    if self.allocation_is_valid(
                        request.kind,
                        &device_indexes,
                        allocation,
                        request.core_selection,
                    ) {
                        self.increment(allocation);
                        return Ok(allocation);
                    }
                }
            }
        }

        let candidates = device_indexes
            .iter()
            .flat_map(|&device_index| {
                let device = &self.devices[device_index];
                device
                    .cores
                    .iter()
                    .enumerate()
                    .filter(move |(_, core)| request.core_selection.contains(core.id))
                    .filter(move |(_, core)| {
                        request.core_selection.is_explicit()
                            || request.mode == SchedulingMode::Auto
                            || request.core_selection.contains(core.id)
                    })
                    .map(move |(core_index, core)| {
                        (device_index, core_index, core.load, device.id, core.id)
                    })
            })
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            return match request.core_selection {
                CoreSelection::Mask(mask) if mask == 0 => Err(Error::InvalidCoreMask { mask }),
                CoreSelection::Mask(mask) => Err(Error::InvalidCoreMask { mask }),
                CoreSelection::Single(core_id) => Err(Error::MissingCore {
                    mask: 1u32 << u32::from(core_id),
                    core_id,
                }),
                _ => Err(Error::NoAvailableCore),
            };
        }

        let selected = candidates
            .into_iter()
            .min_by_key(|(_, _, load, device_id, core_id)| (*load, *device_id, *core_id))
            .expect("candidates not empty");
        let allocation = Allocation {
            device_index: selected.0,
            core_index: selected.1,
        };
        self.increment(allocation);

        if let Some(key) = request.affinity_key {
            self.affinity.insert(key, allocation);
        }

        Ok(allocation)
    }

    fn allocation_is_valid(
        &self,
        kind: DeviceKind,
        device_indexes: &[usize],
        allocation: Allocation,
        selection: CoreSelection,
    ) -> bool {
        let Some(device) = self.devices.get(allocation.device_index) else {
            return false;
        };
        if device.kind != kind || !device_indexes.contains(&allocation.device_index) {
            return false;
        }
        let Some(core) = device.cores.get(allocation.core_index) else {
            return false;
        };
        selection.contains(core.id)
    }

    fn increment(&mut self, allocation: Allocation) {
        let core = &mut self.devices[allocation.device_index].cores[allocation.core_index];
        core.load = core.load.saturating_add(1);
    }

    fn release(&mut self, allocation: Allocation) {
        let core = &mut self.devices[allocation.device_index].cores[allocation.core_index];
        core.load = core.load.saturating_sub(1);
    }
}

fn validate_topology(devices: &[Device]) -> Result<()> {
    if devices.is_empty() {
        return Err(Error::EmptyTopology);
    }

    let mut seen = HashMap::new();
    for device in devices {
        if seen.insert((device.kind, device.id), ()).is_some() {
            return Err(Error::DuplicateDevice {
                kind: device.kind,
                id: device.id,
            });
        }
        if device.cores.is_empty() {
            return Err(Error::EmptyDevice {
                kind: device.kind,
                id: device.id,
            });
        }
        let mut core_ids = HashMap::new();
        for core in &device.cores {
            if core.id >= 32 {
                return Err(Error::CoreIdOutOfRange {
                    kind: device.kind,
                    id: device.id,
                    core_id: core.id,
                });
            }
            if core_ids.insert(core.id, ()).is_some() {
                return Err(Error::DuplicateCore {
                    kind: device.kind,
                    id: device.id,
                    core_id: core.id,
                });
            }
        }
    }
    Ok(())
}

fn cores_from_count(core_count: u8) -> Vec<Core> {
    (0..core_count).map(|id| Core { id }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    fn rknn_topology() -> Scheduler {
        Scheduler::new(
            Topology::multi_card_multi_core(DeviceKind::RknnNpu, [(0, 3), (1, 2)])
                .expect("topology"),
        )
        .expect("scheduler")
    }

    #[test]
    fn least_loaded_spreads_across_cores() {
        let scheduler = rknn_topology();
        let lease_a = scheduler
            .acquire(Request::auto(DeviceKind::RknnNpu))
            .expect("lease a");
        let lease_b = scheduler
            .acquire(Request::auto(DeviceKind::RknnNpu))
            .expect("lease b");
        let snapshot = scheduler.snapshot().expect("snapshot");
        let loads = snapshot
            .iter()
            .flat_map(|device| device.cores.iter().map(|core| core.load))
            .collect::<Vec<_>>();
        assert_eq!(loads.iter().copied().sum::<usize>(), 2);
        assert!(loads.iter().filter(|&&load| load == 1).count() >= 2);
        drop(lease_a);
        drop(lease_b);
        assert!(scheduler
            .snapshot()
            .expect("snapshot")
            .iter()
            .all(|device| device.cores.iter().all(|core| core.load == 0)));
    }

    #[test]
    fn affinity_prefers_previous_core() {
        let scheduler = rknn_topology();
        let lease_a = scheduler
            .acquire(Request::auto(DeviceKind::RknnNpu).with_affinity_key("stream-a"))
            .expect("lease a");
        let first = lease_a.core_id();
        drop(lease_a);
        let lease_b = scheduler
            .acquire(Request::auto(DeviceKind::RknnNpu).with_affinity_key("stream-a"))
            .expect("lease b");
        assert_eq!(lease_b.core_id(), first);
    }

    #[test]
    fn explicit_mask_is_respected() {
        let scheduler = rknn_topology();
        let lease = scheduler
            .acquire(Request::explicit(
                DeviceKind::RknnNpu,
                0,
                CoreSelection::Mask(0b010),
            ))
            .expect("lease");
        assert_eq!(lease.device(), (DeviceKind::RknnNpu, 0));
        assert_eq!(lease.core_id(), 1);
    }

    #[test]
    fn invalid_mask_and_empty_topology_error_cleanly() {
        let err = Topology::new(DeployMode::Host, Vec::new()).expect_err("empty");
        assert!(matches!(err, Error::EmptyTopology));

        let scheduler = rknn_topology();
        let err = scheduler
            .acquire(Request::explicit(
                DeviceKind::RknnNpu,
                0,
                CoreSelection::Mask(0b1000),
            ))
            .expect_err("invalid mask");
        assert!(matches!(err, Error::InvalidCoreMask { mask } if mask == 0b1000));
    }

    #[test]
    fn multi_card_selection_prefers_least_loaded_device() {
        let scheduler = Scheduler::new(
            Topology::multi_card_multi_core(DeviceKind::SophonTpu, [(7, 1), (8, 1)])
                .expect("topology"),
        )
        .expect("scheduler");
        let lease_a = scheduler
            .acquire(Request::auto(DeviceKind::SophonTpu))
            .expect("lease a");
        let lease_b = scheduler
            .acquire(Request::auto(DeviceKind::SophonTpu))
            .expect("lease b");
        assert_ne!(lease_a.device(), lease_b.device());
    }

    #[test]
    fn no_matching_device_is_reported() {
        let scheduler =
            Scheduler::new(Topology::single_chip(DeviceKind::IntelGpu, 1).expect("topology"))
                .expect("scheduler");
        let err = scheduler
            .acquire(Request::auto(DeviceKind::CudaGpu))
            .expect_err("no device");
        assert!(matches!(err, Error::NoMatchingDevice { kind } if kind == DeviceKind::CudaGpu));
    }

    proptest! {
        #[test]
        fn allocation_respects_mask_and_lease_drops_restore_load(mask in 1u32..(1u32 << 3), count in 1usize..8) {
            let scheduler = Scheduler::new(
                Topology::single_chip(DeviceKind::RknnNpu, 3).expect("topology"),
            )
            .expect("scheduler");
            let mut leases = Vec::new();
            for _ in 0..count {
                let lease = scheduler
                    .acquire(Request::explicit(DeviceKind::RknnNpu, 0, CoreSelection::Mask(mask)))
                    .expect("lease");
                prop_assert!(mask & (1u32 << u32::from(lease.core_id())) != 0);
                leases.push(lease);
            }
            let total_load: usize = scheduler
                .snapshot()
                .expect("snapshot")
                .iter()
                .flat_map(|device| device.cores.iter().map(|core| core.load))
                .sum();
            prop_assert_eq!(total_load, count);
            drop(leases);
            let total_load_after_drop: usize = scheduler
                .snapshot()
                .expect("snapshot")
                .iter()
                .flat_map(|device| device.cores.iter().map(|core| core.load))
                .sum();
            prop_assert_eq!(total_load_after_drop, 0);
        }
    }
}
