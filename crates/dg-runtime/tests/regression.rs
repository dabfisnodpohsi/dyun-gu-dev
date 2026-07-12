use dg_runtime::{
    configure_backend, BackendConfig, BackendKind, RegressionCase, RegressionError,
    RegressionHarness, Runtime,
};
use serde_json::json;

fn mock_runtime() -> Runtime {
    let option = configure_backend(
        "mock",
        BackendConfig::new(
            None,
            json!({
                "shape": [1, 4],
                "echo_inputs": true
            }),
        ),
    )
    .expect("configure mock regression runtime");
    assert_eq!(option.backend, BackendKind::Mock);
    Runtime::new(option).expect("create mock regression runtime")
}

#[test]
fn mock_fixture_runs_through_regression_harness() {
    let case = RegressionCase::from_json(include_str!("fixtures/mock_identity_f32.json"))
        .expect("load regression fixture");
    let mut runtime = mock_runtime();
    let report = RegressionHarness::run(&mut runtime, &case).expect("run regression case");
    assert_eq!(report.case, "mock_identity_f32");
    assert_eq!(report.max_absolute_error, 0.0);
    assert_eq!(report.max_relative_error, 0.0);
    assert!((report.minimum_cosine_similarity - 1.0).abs() < f32::EPSILON);
}

#[test]
fn regression_harness_reports_tensor_context_for_mismatch() {
    let mut case = RegressionCase::from_json(include_str!("fixtures/mock_identity_f32.json"))
        .expect("load regression fixture");
    case.expected_outputs[0].values[2] = 3.5;
    case.tolerance.absolute = 0.000001;
    case.tolerance.relative = 0.000001;
    case.tolerance.cosine = None;
    let mut runtime = mock_runtime();
    let error = RegressionHarness::run(&mut runtime, &case).expect_err("mismatch should fail");
    let RegressionError::Mismatch {
        case,
        output,
        index,
        expected,
        actual,
        ..
    } = error
    else {
        panic!("expected detailed mismatch error");
    };
    assert_eq!(case, "mock_identity_f32");
    assert_eq!(output, 0);
    assert_eq!(index, 2);
    assert_eq!(expected, 3.5);
    assert_eq!(actual, 3.25);
}

#[test]
fn regression_case_rejects_invalid_shape_and_tolerance() {
    let mut case = RegressionCase::from_json(include_str!("fixtures/mock_identity_f32.json"))
        .expect("load regression fixture");
    case.inputs[0].shape = vec![1, 3];
    let mut runtime = mock_runtime();
    let error = RegressionHarness::run(&mut runtime, &case).expect_err("shape should fail");
    assert!(matches!(error, RegressionError::Invalid(_)));

    let mut case = RegressionCase::from_json(include_str!("fixtures/mock_identity_f32.json"))
        .expect("load regression fixture");
    case.tolerance.cosine = Some(2.0);
    let error = RegressionHarness::run(&mut runtime, &case).expect_err("cosine should fail");
    assert!(matches!(error, RegressionError::Invalid(_)));
}
