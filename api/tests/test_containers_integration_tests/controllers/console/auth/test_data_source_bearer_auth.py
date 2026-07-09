"""Controller integration tests for API key data source auth routes."""

import json
from unittest.mock import ANY, patch

from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from models.source import DataSourceApiKeyAuthBinding
from tests.test_containers_integration_tests.controllers.console.helpers import (
    authenticate_console_client,
    create_console_account_and_tenant,
)
from tests.test_containers_integration_tests.transactional import DatabaseState


def test_get_api_key_auth_data_source(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
) -> None:
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    foreign_account, foreign_tenant = create_console_account_and_tenant(transactional_db_session)
    binding = DataSourceApiKeyAuthBinding(
        tenant_id=tenant.id,
        category="api_key",
        provider="custom_provider",
        credentials=json.dumps({"auth_type": "api_key", "config": {"api_key": "encrypted"}}),
        disabled=False,
    )
    foreign_binding = DataSourceApiKeyAuthBinding(
        tenant_id=foreign_tenant.id,
        category="api_key",
        provider="foreign_provider",
        credentials=json.dumps({"auth_type": "api_key", "config": {"api_key": "encrypted"}}),
        disabled=False,
    )
    transactional_db_session.add_all([binding, foreign_binding])
    transactional_db_session.commit()
    authenticate_console_client(test_client_with_containers, foreign_account)

    response = test_client_with_containers.get(
        "/console/api/api-key-auth/data-source",
        headers=authenticate_console_client(test_client_with_containers, account),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["provider"] == "custom_provider"


def test_get_api_key_auth_data_source_empty(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
) -> None:
    account, _tenant = create_console_account_and_tenant(transactional_db_session)

    response = test_client_with_containers.get(
        "/console/api/api-key-auth/data-source",
        headers=authenticate_console_client(test_client_with_containers, account),
    )

    assert response.status_code == 200
    assert response.get_json() == {"sources": []}


def test_create_binding_successful(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
) -> None:
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    tenant_id = tenant.id
    payload = {"category": "api_key", "provider": "custom", "credentials": {"key": "value"}}

    with (
        patch("controllers.console.auth.data_source_bearer_auth.ApiKeyAuthService.validate_api_key_auth_args"),
        patch("controllers.console.auth.data_source_bearer_auth.ApiKeyAuthService.create_provider_auth") as create_auth,
    ):
        response = test_client_with_containers.post(
            "/console/api/api-key-auth/data-source/binding",
            json=payload,
            headers=authenticate_console_client(test_client_with_containers, account),
        )

    assert response.status_code == 200
    assert response.get_json() == {"result": "success"}
    create_auth.assert_called_once_with(tenant_id, payload, session=ANY)


def test_create_binding_failure(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
) -> None:
    account, _tenant = create_console_account_and_tenant(transactional_db_session)

    with (
        patch("controllers.console.auth.data_source_bearer_auth.ApiKeyAuthService.validate_api_key_auth_args"),
        patch(
            "controllers.console.auth.data_source_bearer_auth.ApiKeyAuthService.create_provider_auth",
            side_effect=ValueError("Invalid structure"),
        ),
    ):
        response = test_client_with_containers.post(
            "/console/api/api-key-auth/data-source/binding",
            json={"category": "api_key", "provider": "custom", "credentials": {"key": "value"}},
            headers=authenticate_console_client(test_client_with_containers, account),
        )

    assert response.status_code == 500
    payload = response.get_json()
    assert payload is not None
    assert payload["code"] == "auth_failed"
    assert payload["message"] == "Invalid structure"


def test_delete_binding_successful(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
    database_state: DatabaseState,
) -> None:
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    binding = DataSourceApiKeyAuthBinding(
        tenant_id=tenant.id,
        category="api_key",
        provider="custom_provider",
        credentials=json.dumps({"auth_type": "api_key", "config": {"api_key": "encrypted"}}),
        disabled=False,
    )
    transactional_db_session.add(binding)
    transactional_db_session.commit()

    response = test_client_with_containers.delete(
        f"/console/api/api-key-auth/data-source/{binding.id}",
        headers=authenticate_console_client(test_client_with_containers, account),
    )

    assert response.status_code == 204
    assert (
        database_state.count(DataSourceApiKeyAuthBinding, DataSourceApiKeyAuthBinding.id == binding.id) == 0
    )


def test_delete_binding_scopes_to_authenticated_tenant(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
    database_state: DatabaseState,
) -> None:
    account, _tenant = create_console_account_and_tenant(transactional_db_session)
    foreign_account, foreign_tenant = create_console_account_and_tenant(transactional_db_session)
    foreign_binding = DataSourceApiKeyAuthBinding(
        tenant_id=foreign_tenant.id,
        category="api_key",
        provider="custom_provider",
        credentials=json.dumps({"auth_type": "api_key", "config": {"api_key": "encrypted"}}),
        disabled=False,
    )
    transactional_db_session.add(foreign_binding)
    transactional_db_session.commit()
    foreign_binding_id = foreign_binding.id
    authenticate_console_client(test_client_with_containers, foreign_account)

    response = test_client_with_containers.delete(
        f"/console/api/api-key-auth/data-source/{foreign_binding_id}",
        headers=authenticate_console_client(test_client_with_containers, account),
    )

    assert response.status_code == 204
    assert (
        database_state.one(DataSourceApiKeyAuthBinding, DataSourceApiKeyAuthBinding.id == foreign_binding_id).id
        == foreign_binding_id
    )
