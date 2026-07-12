use std::future::Future;
use std::sync::Arc;
use std::thread;

use async_trait::async_trait;

use crate::connector::{install_cheetah_connector, CheetahRuntimeConnector, StreamProtocol};
use crate::error::{Error, Result};
use crate::stream::{PublisherOptions, SubscriberOptions};

type CheetahAvFrame = dg_stream_cheetah::AVFrame;
type CheetahConnectorError = dg_stream_cheetah::cheetah_connector::ConnectorError;
type CheetahPullHandle = dg_stream_cheetah::cheetah_connector::PullHandle;
type CheetahPushHandle = dg_stream_cheetah::cheetah_connector::PushHandle;
type CheetahSubscriberSource = dyn dg_stream_cheetah::SubscriberSource;
type CheetahPublisherSink = dyn dg_stream_cheetah::PublisherSink;

struct RuntimeContext {
    connector: dg_stream_cheetah::cheetah_connector::EngineConnector,
    runtime: Option<Arc<tokio::runtime::Runtime>>,
}

impl RuntimeContext {
    fn new() -> Result<Arc<Self>> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|err| Error::Runtime(format!("build Tokio runtime: {err}")))?,
        );
        let runtime_api = Arc::new(dg_stream_cheetah::cheetah_runtime_tokio::TokioRuntime::new())
            as Arc<dyn dg_stream_cheetah::RuntimeApi>;
        let connector = dg_stream_cheetah::cheetah_connector::ConnectorBuilder::new(runtime_api)
            .with_default_modules()
            .build()
            .map_err(map_connector_error)?;
        let connector = run_on_runtime(runtime.clone(), async move {
            connector
                .start()
                .await
                .map(|()| connector)
                .map_err(map_connector_error)
        })?;
        Ok(Arc::new(Self {
            connector,
            runtime: Some(runtime),
        }))
    }

    fn run<F, T>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Error::Runtime("embedded Tokio runtime is stopped".to_string()))?;
        run_on_runtime(Arc::clone(runtime), future)
    }
}

impl Drop for RuntimeContext {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            let _ = thread::Builder::new()
                .name("dg-stream-cheetah-shutdown".to_string())
                .spawn(move || drop(runtime));
        } else {
            drop(runtime);
        }
    }
}

fn run_on_runtime<F, T>(runtime: Arc<tokio::runtime::Runtime>, future: F) -> Result<T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        thread::Builder::new()
            .name("dg-stream-cheetah-open".to_string())
            .spawn(move || runtime.block_on(future))
            .map_err(|err| Error::Runtime(format!("spawn Tokio bridge thread: {err}")))?
            .join()
            .map_err(|_| Error::Runtime("Tokio bridge thread panicked".to_string()))?
    } else {
        runtime.block_on(future)
    }
}

fn map_connector_error(err: CheetahConnectorError) -> Error {
    Error::Sdk(err.to_string())
}

fn map_protocol(protocol: StreamProtocol) -> dg_stream_cheetah::cheetah_connector::Protocol {
    match protocol {
        StreamProtocol::RtspPull => dg_stream_cheetah::cheetah_connector::Protocol::Rtsp,
        StreamProtocol::HttpFlvPull => dg_stream_cheetah::cheetah_connector::Protocol::HttpFlv,
        StreamProtocol::RtmpPush => dg_stream_cheetah::cheetah_connector::Protocol::Rtmp,
        StreamProtocol::WebRtcPush => dg_stream_cheetah::cheetah_connector::Protocol::WebRtc,
    }
}

fn map_subscriber_options(options: SubscriberOptions) -> dg_stream_cheetah::SubscriberOptions {
    dg_stream_cheetah::SubscriberOptions {
        queue_capacity: options.queue_capacity,
        backpressure: match options.backpressure {
            crate::stream::BackpressurePolicy::DropDroppableFirst => {
                dg_stream_cheetah::BackpressurePolicy::DropDroppableFirst
            }
            crate::stream::BackpressurePolicy::DropUntilNextKeyframe => {
                dg_stream_cheetah::BackpressurePolicy::DropUntilNextKeyframe
            }
            crate::stream::BackpressurePolicy::DisconnectOnOverflow => {
                dg_stream_cheetah::BackpressurePolicy::DisconnectOnOverflow
            }
        },
        bootstrap_policy: dg_stream_cheetah::BootstrapPolicy {
            mode: match options.bootstrap_policy.mode {
                crate::stream::BootstrapMode::None => dg_stream_cheetah::BootstrapMode::None,
                crate::stream::BootstrapMode::LiveTail => {
                    dg_stream_cheetah::BootstrapMode::LiveTail
                }
                crate::stream::BootstrapMode::FullGop => dg_stream_cheetah::BootstrapMode::FullGop,
            },
            max_bootstrap_age_ms: options.bootstrap_policy.max_bootstrap_age_ms,
            max_bootstrap_frames: options.bootstrap_policy.max_bootstrap_frames,
            wait_for_next_random_access_point: options
                .bootstrap_policy
                .wait_for_next_random_access_point,
        },
        media_filter: dg_stream_cheetah::MediaFilter {
            enable_video: options.media_filter.enable_video,
            enable_audio: options.media_filter.enable_audio,
        },
    }
}

fn map_publisher_options(options: PublisherOptions) -> dg_stream_cheetah::PublisherOptions {
    dg_stream_cheetah::PublisherOptions {
        announce_tracks: options.announce_tracks,
    }
}

struct RuntimeOwnedPull {
    inner: CheetahPullHandle,
    _context: Arc<RuntimeContext>,
}

#[async_trait]
impl dg_stream_cheetah::SubscriberSource for RuntimeOwnedPull {
    async fn recv(
        &mut self,
    ) -> std::result::Result<Option<Arc<CheetahAvFrame>>, dg_stream_cheetah::SdkError> {
        dg_stream_cheetah::SubscriberSource::recv(&mut self.inner).await
    }

    async fn close(&mut self) -> std::result::Result<(), dg_stream_cheetah::SdkError> {
        dg_stream_cheetah::SubscriberSource::close(&mut self.inner).await
    }

    fn id(&self) -> dg_stream_cheetah::SubscriberId {
        dg_stream_cheetah::SubscriberSource::id(&self.inner)
    }

    fn tracks(&self) -> Vec<dg_stream_cheetah::TrackInfo> {
        dg_stream_cheetah::SubscriberSource::tracks(&self.inner)
    }
}

struct RuntimeOwnedPush {
    inner: CheetahPushHandle,
    _context: Arc<RuntimeContext>,
}

impl dg_stream_cheetah::PublisherSink for RuntimeOwnedPush {
    fn update_tracks(
        &self,
        tracks: Vec<dg_stream_cheetah::TrackInfo>,
    ) -> std::result::Result<(), dg_stream_cheetah::SdkError> {
        dg_stream_cheetah::PublisherSink::update_tracks(&self.inner, tracks)
    }

    fn push_frame(
        &self,
        frame: Arc<CheetahAvFrame>,
    ) -> std::result::Result<dg_stream_cheetah::DispatchResult, dg_stream_cheetah::SdkError> {
        dg_stream_cheetah::PublisherSink::push_frame(&self.inner, frame)
    }

    fn close(&self) -> std::result::Result<(), dg_stream_cheetah::SdkError> {
        dg_stream_cheetah::PublisherSink::close(&self.inner)
    }

    fn take_keyframe_requests(&self) -> u64 {
        dg_stream_cheetah::PublisherSink::take_keyframe_requests(&self.inner)
    }
}

/// Embedded runtime-backed connector for the first-party Cheetah adapters.
pub struct EmbeddedCheetahRuntimeConnector {
    context: Arc<RuntimeContext>,
}

impl EmbeddedCheetahRuntimeConnector {
    /// Builds and starts an embedded Cheetah engine with all connector modules.
    pub fn new() -> Result<Self> {
        Ok(Self {
            context: RuntimeContext::new()?,
        })
    }
}

impl CheetahRuntimeConnector for EmbeddedCheetahRuntimeConnector {
    fn open_pull(
        &self,
        protocol: StreamProtocol,
        url: &str,
        options: SubscriberOptions,
    ) -> Result<(Vec<crate::track::TrackInfo>, Box<CheetahSubscriberSource>)> {
        let protocol = map_protocol(protocol);
        let url = url.to_string();
        let subscriber = map_subscriber_options(options);
        let context = Arc::clone(&self.context);
        let operation_context = Arc::clone(&context);
        let (tracks, handle) = context.run(async move {
            let options = dg_stream_cheetah::cheetah_connector::ConnectorPullOptions {
                subscriber,
                ..Default::default()
            };
            let handle = dg_stream_cheetah::cheetah_connector::RuntimeConnector::open_pull(
                &operation_context.connector,
                protocol,
                &url,
                options,
            )
            .await
            .map_err(map_connector_error)?;
            let tracks = dg_stream_cheetah::SubscriberSource::tracks(&handle)
                .into_iter()
                .map(|track| crate::bridge::cheetah_track_info_to_media_frame(&track))
                .collect();
            Ok((tracks, handle))
        })?;
        Ok((
            tracks,
            Box::new(RuntimeOwnedPull {
                inner: handle,
                _context: context,
            }),
        ))
    }

    fn open_push(
        &self,
        protocol: StreamProtocol,
        url: &str,
        options: PublisherOptions,
    ) -> Result<Box<CheetahPublisherSink>> {
        let protocol = map_protocol(protocol);
        let url = url.to_string();
        let publisher = map_publisher_options(options);
        let context = Arc::clone(&self.context);
        let operation_context = Arc::clone(&context);
        let handle = context.run(async move {
            let options = dg_stream_cheetah::cheetah_connector::ConnectorPushOptions {
                publisher,
                ..Default::default()
            };
            dg_stream_cheetah::cheetah_connector::RuntimeConnector::open_push(
                &operation_context.connector,
                protocol,
                &url,
                options,
            )
            .await
            .map_err(map_connector_error)
        })?;
        Ok(Box::new(RuntimeOwnedPush {
            inner: handle,
            _context: context,
        }))
    }
}

/// Installs a newly constructed embedded connector as the process-wide connector.
pub fn install_embedded_cheetah_connector() -> Result<()> {
    install_cheetah_connector(Box::new(EmbeddedCheetahRuntimeConnector::new()?))
}
