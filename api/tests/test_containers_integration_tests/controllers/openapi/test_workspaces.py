from __future__ import annotations

from collections.abc import Callable

import pytest
from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from models import Account
from models.account import TenantAccountRole
from tests.test_containers_integration_tests.controllers.openapi.conftest import BearerFactory, add_tenant_for_account

pytestmark = pytest.mark.requires_redis


class TestWorkspacesList:
    def test_lists_only_members_workspaces_with_role(
        self,
        test_client_with_containers: FlaskClient,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        owner_tenant = account.current_tenant
        assert owner_tenant is not None
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get("/openapi/v1/workspaces", headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        ids = {workspace["id"] for workspace in result["workspaces"]}
        assert ids == {owner_tenant.id}
        only = result["workspaces"][0]
        assert only["role"] == TenantAccountRole.OWNER.value
        assert only["status"] == "normal"
        assert only["current"] is False

    def test_lists_all_joined_workspaces(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        owner_tenant = account.current_tenant
        assert owner_tenant is not None
        second = add_tenant_for_account(account, session=transactional_db_session, role="normal", name="Second WS")
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get("/openapi/v1/workspaces", headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        assert {workspace["id"] for workspace in result["workspaces"]} == {owner_tenant.id, second.id}


class TestWorkspaceDetail:
    def test_member_can_read_detail(
        self,
        test_client_with_containers: FlaskClient,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        tenant = account.current_tenant
        assert tenant is not None
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(f"/openapi/v1/workspaces/{tenant.id}", headers=headers)

        assert response.status_code == 200
        detail = response.get_json()
        assert detail["id"] == tenant.id
        assert detail["role"] == TenantAccountRole.OWNER.value
        assert detail["current"] is False
        assert detail["created_at"] is not None

    def test_non_member_detail_is_404_not_403(
        self,
        test_client_with_containers: FlaskClient,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        owner = make_transactional_account()
        outsider = make_transactional_account()
        someone_elses_ws = owner.current_tenant
        assert someone_elses_ws is not None
        headers, _mint = account_bearer_factory(outsider)

        response = test_client_with_containers.get(
            f"/openapi/v1/workspaces/{someone_elses_ws.id}", headers=headers
        )

        assert response.status_code == 404
        assert response.get_json()["message"] == "workspace not found"


class TestWorkspaceSwitch:
    def test_switch_sets_current_and_persists(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        owner_tenant = account.current_tenant
        assert owner_tenant is not None
        target = add_tenant_for_account(
            account, session=transactional_db_session, role="normal", name="Switch Target"
        )
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.post(
            f"/openapi/v1/workspaces/{target.id}:switch", headers=headers
        )

        assert response.status_code == 200
        detail = response.get_json()
        assert detail["id"] == target.id
        assert detail["current"] is True

        listing_response = test_client_with_containers.get("/openapi/v1/workspaces", headers=headers)
        assert listing_response.status_code == 200
        by_id = {workspace["id"]: workspace for workspace in listing_response.get_json()["workspaces"]}
        assert by_id[target.id]["current"] is True
        assert by_id[owner_tenant.id]["current"] is False

    def test_switch_to_non_member_workspace_is_404(
        self,
        test_client_with_containers: FlaskClient,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        outsider_ws = make_transactional_account().current_tenant
        assert outsider_ws is not None
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.post(
            f"/openapi/v1/workspaces/{outsider_ws.id}:switch", headers=headers
        )

        assert response.status_code == 404
        assert response.get_json() is not None
