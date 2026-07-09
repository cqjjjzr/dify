"""Integration tests for console API key endpoints using testcontainers."""

from __future__ import annotations

import pytest
from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from models.account import TenantAccountRole
from models.enums import ApiTokenType
from models.model import ApiToken, App, AppMode
from tests.test_containers_integration_tests.controllers.console.helpers import (
    ConsoleAccountFactory,
    authenticate_console_client,
    create_console_account_and_tenant,
    create_console_app,
)
from tests.test_containers_integration_tests.transactional import DatabaseState


@pytest.fixture
def setup_app(
    transactional_db_session: Session,
    test_client_with_containers: FlaskClient,
) -> tuple[FlaskClient, dict[str, str], App]:
    """Create an authenticated client with an app for API key tests."""
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    app = create_console_app(transactional_db_session, tenant.id, account.id, AppMode.CHAT)
    headers = authenticate_console_client(test_client_with_containers, account)
    return test_client_with_containers, headers, app


class TestAppApiKeyListResource:
    """Tests for GET/POST /apps/<resource_id>/api-keys."""

    def test_get_empty_keys(self, setup_app: tuple[FlaskClient, dict[str, str], App]) -> None:
        client, headers, app = setup_app
        resp = client.get(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        assert resp.status_code == 200
        assert resp.json is not None
        assert resp.json["data"] == []

    def test_create_api_key(self, setup_app: tuple[FlaskClient, dict[str, str], App]) -> None:
        client, headers, app = setup_app
        resp = client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        assert resp.status_code == 201
        data = resp.json
        assert data is not None
        assert data["token"].startswith("app-")
        assert data["id"] is not None

    def test_create_api_key_persists_authenticated_tenant(
        self,
        setup_app: tuple[FlaskClient, dict[str, str], App],
        database_state: DatabaseState,
    ) -> None:
        client, headers, app = setup_app
        tenant_id = app.tenant_id

        with database_state.expect_count_change(ApiToken, ApiToken.app_id == app.id, before=0, after=1):
            resp = client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)
            assert resp.status_code == 201

        assert resp.json is not None
        api_token = database_state.one(ApiToken, ApiToken.id == resp.json["id"])
        assert api_token.tenant_id == tenant_id
        assert api_token.app_id == app.id
        assert api_token.type == ApiTokenType.APP

    def test_get_keys_after_create(self, setup_app: tuple[FlaskClient, dict[str, str], App]) -> None:
        client, headers, app = setup_app
        client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)

        resp = client.get(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        assert resp.status_code == 200
        assert resp.json is not None
        assert len(resp.json["data"]) == 2

    def test_create_key_max_limit(
        self,
        setup_app: tuple[FlaskClient, dict[str, str], App],
    ) -> None:
        client, headers, app = setup_app
        # Create 10 keys (the max)
        for _ in range(10):
            client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)

        # 11th should fail
        resp = client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        assert resp.status_code == 400

    def test_get_keys_for_nonexistent_app(
        self,
        setup_app: tuple[FlaskClient, dict[str, str], App],
    ) -> None:
        client, headers, _ = setup_app
        resp = client.get(
            "/console/api/apps/00000000-0000-0000-0000-000000000000/api-keys",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_get_foreign_app_keys_not_found(
        self,
        setup_app: tuple[FlaskClient, dict[str, str], App],
        transactional_db_session: Session,
    ) -> None:
        client, headers, _ = setup_app
        foreign_account, foreign_tenant = create_console_account_and_tenant(transactional_db_session)
        foreign_app = create_console_app(
            transactional_db_session, foreign_tenant.id, foreign_account.id, AppMode.CHAT
        )

        resp = client.get(f"/console/api/apps/{foreign_app.id}/api-keys", headers=headers)

        assert resp.status_code == 404


class TestAppApiKeyResource:
    """Tests for DELETE /apps/<resource_id>/api-keys/<api_key_id>."""

    @pytest.mark.requires_redis
    def test_delete_key_success(self, setup_app: tuple[FlaskClient, dict[str, str], App]) -> None:
        client, headers, app = setup_app
        create_resp = client.post(f"/console/api/apps/{app.id}/api-keys", headers=headers)
        assert create_resp.json is not None
        key_id = create_resp.json["id"]

        resp = client.delete(f"/console/api/apps/{app.id}/api-keys/{key_id}", headers=headers)
        assert resp.status_code == 204

    def test_delete_nonexistent_key(self, setup_app: tuple[FlaskClient, dict[str, str], App]) -> None:
        client, headers, app = setup_app
        resp = client.delete(
            f"/console/api/apps/{app.id}/api-keys/00000000-0000-0000-0000-000000000000",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_delete_key_nonexistent_app(
        self,
        setup_app: tuple[FlaskClient, dict[str, str], App],
    ) -> None:
        client, headers, _ = setup_app
        resp = client.delete(
            "/console/api/apps/00000000-0000-0000-0000-000000000000/api-keys/00000000-0000-0000-0000-000000000000",
            headers=headers,
        )
        assert resp.status_code == 404

    def test_delete_forbidden_for_non_admin(
        self,
        console_account_factory: ConsoleAccountFactory,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        database_state: DatabaseState,
    ) -> None:
        account, tenant = console_account_factory(role=TenantAccountRole.NORMAL)
        app = create_console_app(transactional_db_session, tenant.id, account.id, AppMode.CHAT)
        api_token = ApiToken(
            app_id=app.id,
            tenant_id=tenant.id,
            token=ApiToken.generate_api_key("app-", 24),
            type=ApiTokenType.APP,
        )
        transactional_db_session.add(api_token)
        transactional_db_session.commit()
        api_token_id = api_token.id
        headers = authenticate_console_client(test_client_with_containers, account)

        response = test_client_with_containers.delete(
            f"/console/api/apps/{app.id}/api-keys/{api_token_id}",
            headers=headers,
        )

        assert response.status_code == 403
        assert database_state.one(ApiToken, ApiToken.id == api_token_id).id == api_token_id
