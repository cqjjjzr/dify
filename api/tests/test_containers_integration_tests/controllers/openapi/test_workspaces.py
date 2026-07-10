from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch
from uuid import uuid4

import pytest
from flask.testing import FlaskClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import Account, TenantAccountJoin
from models.account import AccountStatus, TenantAccountRole
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


class TestWorkspaceMembers:
    def test_list_update_and_delete_member_persist(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        owner = make_transactional_account()
        tenant = owner.current_tenant
        assert tenant is not None
        member = make_transactional_account()
        tenant_id = tenant.id
        owner_id = owner.id
        member_id = member.id
        transactional_db_session.add(
            TenantAccountJoin(
                tenant_id=tenant_id,
                account_id=member_id,
                role=TenantAccountRole.NORMAL,
                current=False,
            )
        )
        transactional_db_session.commit()
        headers, _mint = account_bearer_factory(owner)
        members_url = f"/openapi/v1/workspaces/{tenant_id}/members"

        list_response = test_client_with_containers.get(members_url, headers=headers)
        assert list_response.status_code == 200
        listed_ids = {item["id"] for item in list_response.get_json()["data"]}
        assert {owner_id, member_id} <= listed_ids

        update_response = test_client_with_containers.put(
            f"{members_url}/{member_id}/role",
            headers=headers,
            json={"role": "admin"},
        )
        assert update_response.status_code == 200
        transactional_db_session.expire_all()
        membership = transactional_db_session.scalars(
            select(TenantAccountJoin).where(
                TenantAccountJoin.tenant_id == tenant_id,
                TenantAccountJoin.account_id == member_id,
            )
        ).one()
        assert membership.role == TenantAccountRole.ADMIN

        delete_response = test_client_with_containers.delete(f"{members_url}/{member_id}", headers=headers)
        assert delete_response.status_code == 200
        remaining = transactional_db_session.scalar(
            select(func.count())
            .select_from(TenantAccountJoin)
            .where(
                TenantAccountJoin.tenant_id == tenant_id,
                TenantAccountJoin.account_id == member_id,
            )
        )
        assert remaining == 0

    def test_invite_member_persists_pending_account(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        owner = make_transactional_account()
        tenant = owner.current_tenant
        assert tenant is not None
        tenant_id = tenant.id
        invitee_email = f"openapi-invite-{uuid4()}@example.com"
        headers, _mint = account_bearer_factory(owner)

        with patch("services.account_service.send_invite_member_mail_task.delay") as send_mail:
            response = test_client_with_containers.post(
                f"/openapi/v1/workspaces/{tenant_id}/members",
                headers=headers,
                json={"email": invitee_email, "role": "normal"},
            )

        assert response.status_code == 201
        payload = response.get_json()
        assert payload["email"] == invitee_email
        invitee = transactional_db_session.scalars(select(Account).where(Account.email == invitee_email)).one()
        assert invitee.status == AccountStatus.PENDING
        membership = transactional_db_session.scalars(
            select(TenantAccountJoin).where(
                TenantAccountJoin.tenant_id == tenant_id,
                TenantAccountJoin.account_id == invitee.id,
            )
        ).one()
        assert membership.role == TenantAccountRole.NORMAL
        send_mail.assert_called_once()
