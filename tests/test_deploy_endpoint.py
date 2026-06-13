from unittest.mock import MagicMock

import botocore
import pytest

from scripts import deploy_endpoint


def _client_error(code: str, message: str, operation_name: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": message}},
        operation_name,
    )


def test_deploy_endpoint_creates_when_missing():
    sm = MagicMock()
    waiter = MagicMock()
    sm.describe_endpoint.side_effect = _client_error(
        "ValidationException",
        "Could not find endpoint",
        "DescribeEndpoint",
    )
    sm.get_waiter.return_value = waiter

    deploy_endpoint.deploy_endpoint(sm, endpoint_name="penguin-endpoint", config_name="cfg-v1")

    sm.create_endpoint.assert_called_once_with(
        EndpointName="penguin-endpoint",
        EndpointConfigName="cfg-v1",
    )
    sm.update_endpoint.assert_not_called()
    sm.delete_endpoint.assert_not_called()
    sm.get_waiter.assert_called_once_with("endpoint_in_service")
    waiter.wait.assert_called_once_with(
        EndpointName="penguin-endpoint",
        WaiterConfig=deploy_endpoint.WAITER_CONFIG,
    )


def test_deploy_endpoint_updates_when_in_service():
    sm = MagicMock()
    waiter = MagicMock()
    sm.describe_endpoint.return_value = {"EndpointStatus": "InService"}
    sm.get_waiter.return_value = waiter

    deploy_endpoint.deploy_endpoint(sm, endpoint_name="penguin-endpoint", config_name="cfg-v2")

    sm.update_endpoint.assert_called_once_with(
        EndpointName="penguin-endpoint",
        EndpointConfigName="cfg-v2",
    )
    sm.create_endpoint.assert_not_called()
    sm.delete_endpoint.assert_not_called()
    waiter.wait.assert_called_once_with(
        EndpointName="penguin-endpoint",
        WaiterConfig=deploy_endpoint.WAITER_CONFIG,
    )


def test_deploy_endpoint_recreates_when_failed(caplog):
    sm = MagicMock()
    delete_waiter = MagicMock()
    service_waiter = MagicMock()
    sm.describe_endpoint.return_value = {
        "EndpointStatus": "Failed",
        "FailureReason": "Container crashed during startup",
    }
    sm.get_waiter.side_effect = lambda name: {
        "endpoint_deleted": delete_waiter,
        "endpoint_in_service": service_waiter,
    }[name]

    with caplog.at_level("INFO"):
        deploy_endpoint.deploy_endpoint(sm, endpoint_name="penguin-endpoint", config_name="cfg-v3")

    sm.delete_endpoint.assert_called_once_with(EndpointName="penguin-endpoint")
    sm.create_endpoint.assert_called_once_with(
        EndpointName="penguin-endpoint",
        EndpointConfigName="cfg-v3",
    )
    sm.update_endpoint.assert_not_called()
    delete_waiter.wait.assert_called_once_with(
        EndpointName="penguin-endpoint",
        WaiterConfig=deploy_endpoint.WAITER_CONFIG,
    )
    service_waiter.wait.assert_called_once_with(
        EndpointName="penguin-endpoint",
        WaiterConfig=deploy_endpoint.WAITER_CONFIG,
    )
    assert "Container crashed during startup" in caplog.text


def test_deploy_endpoint_surfaces_non_validation_error():
    sm = MagicMock()
    sm.describe_endpoint.return_value = {"EndpointStatus": "InService"}
    sm.update_endpoint.side_effect = _client_error(
        "AccessDeniedException",
        "Not authorized",
        "UpdateEndpoint",
    )

    with pytest.raises(botocore.exceptions.ClientError) as exc_info:
        deploy_endpoint.deploy_endpoint(sm, endpoint_name="penguin-endpoint", config_name="cfg-v4")

    assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
    sm.create_endpoint.assert_not_called()
    sm.delete_endpoint.assert_not_called()
