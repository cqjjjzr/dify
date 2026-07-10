from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest
from flask.testing import FlaskClient
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

import services
from controllers.console.workspace import members as members_module
from libs.datetime_utils import naive_utc_now
from models.account import (
    Account,
    AccountStatus,
    Tenant,
    TenantAccountJoin,
    TenantAccountRole,
    TenantStatus,
)
from tests.test_containers_integration_tests.controllers.console.helpers import (
    authenticate_console_client,
    ensure_dify_setup,
)

pytestmark = pytest.mark.requires_redis


class WorkspaceMembersIntegrationFactory:
    @staticmethod
    def create_tenant(transactional_db_session: Session) -> Tenant:
        tenant = Tenant(name=f"Tenant {uuid4()}", plan="basic", status=TenantStatus.NORMAL)
        transactional_db_session.add(tenant)
        transactional_db_session.commit()
        return tenant

    @staticmethod
    def create_account(
        transactional_db_session: Session,
        *,
        email_prefix: str,
        tenant: Tenant | None = None,
        role: TenantAccountRole = TenantAccountRole.NORMAL,
        current: bool = False,
    ) -> Account:
        account = Account(
            name=f"Account {uuid4()}",
            email=f"{email_prefix}-{uuid4()}@example.com",
            password="hashed-password",
            password_salt="salt",
            interface_language="en-US",
            timezone="UTC",
            status=AccountStatus.ACTIVE,
            initialized_at=naive_utc_now(),
        )
        transactional_db_session.add(account)
        transactional_db_session.commit()

        if tenant is not None:
            transactional_db_session.add(
                TenantAccountJoin(
                    tenant_id=tenant.id,
                    account_id=account.id,
                    role=role,
                    current=current,
                )
            )
            transactional_db_session.commit()
            account.current_tenant = tenant
        ensure_dify_setup(transactional_db_session)
        return account

    @staticmethod
    def create_owner_workspace(transactional_db_session: Session) -> tuple[Tenant, Account]:
        tenant = WorkspaceMembersIntegrationFactory.create_tenant(transactional_db_session)
        owner = WorkspaceMembersIntegrationFactory.create_account(
            transactional_db_session,
            email_prefix="owner",
            tenant=tenant,
            role=TenantAccountRole.OWNER,
            current=True,
        )
        return tenant, owner

    @staticmethod
    def create_owner_transfer_token(account: Account) -> str:
        _, token = members_module.AccountService.generate_owner_transfer_token(
            account.email,
            account=account,
            code="123456",
            additional_data={},
        )
        return token

    @staticmethod
    def get_join(transactional_db_session: Session, *, tenant: Tenant, account: Account) -> TenantAccountJoin:
        tenant_id = inspect(tenant).identity[0]
        account_id = inspect(account).identity[0]
        transactional_db_session.expire_all()
        return transactional_db_session.scalars(
            select(TenantAccountJoin).where(
                TenantAccountJoin.tenant_id == tenant_id,
                TenantAccountJoin.account_id == account_id,
            )
        ).one()

    @staticmethod
    def join_count(transactional_db_session: Session, *, tenant: Tenant, account: Account) -> int:
        tenant_id = inspect(tenant).identity[0]
        account_id = inspect(account).identity[0]
        return transactional_db_session.scalar(
            select(func.count())
            .select_from(TenantAccountJoin)
            .where(
                TenantAccountJoin.tenant_id == tenant_id,
                TenantAccountJoin.account_id == account_id,
            )
        ) or 0


def _headers(client: FlaskClient, account: Account) -> dict[str, str]:
    return authenticate_console_client(client, account)


class TestMemberCancelInviteApiWithContainers:
    def test_cancel_success(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(
            transactional_db_session,
            email_prefix="member",
            tenant=tenant,
            role=TenantAccountRole.NORMAL,
        )

        response = test_client_with_containers.delete(
            f"/console/api/workspaces/current/members/{member.id}",
            headers=_headers(test_client_with_containers, current_user),
        )

        assert response.status_code == 200
        assert response.get_json()["result"] == "success"
        assert factory.join_count(transactional_db_session, tenant=tenant, account=member) == 0

    def test_cancel_not_found(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        _tenant, current_user = WorkspaceMembersIntegrationFactory.create_owner_workspace(transactional_db_session)

        response = test_client_with_containers.delete(
            f"/console/api/workspaces/current/members/{uuid4()}",
            headers=_headers(test_client_with_containers, current_user),
        )

        assert response.status_code == 404

    @pytest.mark.parametrize(
        ("error", "status", "code"),
        [
            (services.errors.account.CannotOperateSelfError("x"), 400, "cannot-operate-self"),
            (services.errors.account.NoPermissionError("x"), 403, "forbidden"),
            (services.errors.account.MemberNotInTenantError(), 404, "member-not-found"),
        ],
    )
    def test_cancel_maps_service_errors(
        self,
        error: Exception,
        status: int,
        code: str,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        _tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(transactional_db_session, email_prefix="member")

        with patch.object(members_module.TenantService, "remove_member_from_tenant", side_effect=error):
            response = test_client_with_containers.delete(
                f"/console/api/workspaces/current/members/{member.id}",
                headers=_headers(test_client_with_containers, current_user),
            )

        assert response.status_code == status
        assert response.get_json()["code"] == code


class TestMemberUpdateRoleApiWithContainers:
    def test_update_success(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(
            transactional_db_session,
            email_prefix="member",
            tenant=tenant,
            role=TenantAccountRole.EDITOR,
        )

        response = test_client_with_containers.put(
            f"/console/api/workspaces/current/members/{member.id}/update-role",
            headers=_headers(test_client_with_containers, current_user),
            json={"role": "normal"},
        )

        assert response.status_code == 200
        assert response.get_json()["result"] == "success"
        assert (
            factory.get_join(transactional_db_session, tenant=tenant, account=member).role
            == TenantAccountRole.NORMAL
        )

    def test_update_member_not_found(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        _tenant, current_user = WorkspaceMembersIntegrationFactory.create_owner_workspace(transactional_db_session)

        response = test_client_with_containers.put(
            f"/console/api/workspaces/current/members/{uuid4()}/update-role",
            headers=_headers(test_client_with_containers, current_user),
            json={"role": "normal"},
        )

        assert response.status_code == 404


class TestMemberReadAndInviteApisWithContainers:
    def test_list_members_returns_persisted_memberships(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(
            transactional_db_session,
            email_prefix="listed-member",
            tenant=tenant,
            role=TenantAccountRole.EDITOR,
        )

        response = test_client_with_containers.get(
            "/console/api/workspaces/current/members",
            headers=_headers(test_client_with_containers, current_user),
        )

        assert response.status_code == 200
        accounts = {item["id"]: item for item in response.get_json()["accounts"]}
        assert accounts[current_user.id]["role"] == TenantAccountRole.OWNER.value
        assert accounts[member.id]["role"] == TenantAccountRole.EDITOR.value

    def test_list_dataset_operators_returns_only_operator_members(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        operator = factory.create_account(
            transactional_db_session,
            email_prefix="operator",
            tenant=tenant,
            role=TenantAccountRole.DATASET_OPERATOR,
        )
        operator_id = operator.id
        factory.create_account(
            transactional_db_session,
            email_prefix="ordinary",
            tenant=tenant,
            role=TenantAccountRole.NORMAL,
        )

        response = test_client_with_containers.get(
            "/console/api/workspaces/current/dataset-operators",
            headers=_headers(test_client_with_containers, current_user),
        )

        assert response.status_code == 200
        assert [item["id"] for item in response.get_json()["accounts"]] == [operator_id]

    def test_invite_member_persists_pending_account_and_membership(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        invitee_email = f"invitee-{uuid4()}@example.com"

        with patch("services.account_service.send_invite_member_mail_task.delay") as send_mail:
            response = test_client_with_containers.post(
                "/console/api/workspaces/current/members/invite-email",
                headers=_headers(test_client_with_containers, current_user),
                json={"emails": [invitee_email], "role": "normal", "language": "en-US"},
            )

        assert response.status_code == 201
        result = response.get_json()["invitation_results"][0]
        assert result["status"] == "success"
        invitee = transactional_db_session.scalars(select(Account).where(Account.email == invitee_email)).one()
        assert invitee.status == AccountStatus.PENDING
        assert (
            factory.get_join(transactional_db_session, tenant=tenant, account=invitee).role
            == TenantAccountRole.NORMAL
        )
        send_mail.assert_called_once()


class TestOwnerTransferSupportApisWithContainers:
    def test_send_owner_transfer_email_returns_service_token(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        _tenant, current_user = WorkspaceMembersIntegrationFactory.create_owner_workspace(transactional_db_session)

        with (
            patch.object(members_module.AccountService, "is_email_send_ip_limit", return_value=False),
            patch.object(members_module.AccountService, "send_owner_transfer_email", return_value="transfer-token"),
        ):
            response = test_client_with_containers.post(
                "/console/api/workspaces/current/members/send-owner-transfer-confirm-email",
                headers=_headers(test_client_with_containers, current_user),
                json={"language": "en-US"},
            )

        assert response.status_code == 200
        assert response.get_json() == {"result": "success", "data": "transfer-token"}

    def test_owner_transfer_check_rotates_redis_token(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        _tenant, current_user = WorkspaceMembersIntegrationFactory.create_owner_workspace(transactional_db_session)
        token = WorkspaceMembersIntegrationFactory.create_owner_transfer_token(current_user)

        response = test_client_with_containers.post(
            "/console/api/workspaces/current/members/owner-transfer-check",
            headers=_headers(test_client_with_containers, current_user),
            json={"token": token, "code": "123456"},
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["is_valid"] is True
        assert payload["email"] == current_user.email
        assert members_module.AccountService.get_owner_transfer_data(token) is None
        assert members_module.AccountService.get_owner_transfer_data(payload["token"])["code"] == "123456"


class TestOwnerTransferApiWithContainers:
    def test_member_not_in_tenant(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        _tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(transactional_db_session, email_prefix="member")
        token = factory.create_owner_transfer_token(current_user)

        response = test_client_with_containers.post(
            f"/console/api/workspaces/current/members/{member.id}/owner-transfer",
            headers=_headers(test_client_with_containers, current_user),
            json={"token": token},
        )

        assert response.status_code == 400
        assert response.get_json()["code"] == "member_not_in_tenant"

    def test_member_not_found(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        _tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        token = factory.create_owner_transfer_token(current_user)

        response = test_client_with_containers.post(
            f"/console/api/workspaces/current/members/{uuid4()}/owner-transfer",
            headers=_headers(test_client_with_containers, current_user),
            json={"token": token},
        )

        assert response.status_code == 404

    def test_transfer_success(
        self, test_client_with_containers: FlaskClient, transactional_db_session: Session
    ) -> None:
        factory = WorkspaceMembersIntegrationFactory
        tenant, current_user = factory.create_owner_workspace(transactional_db_session)
        member = factory.create_account(
            transactional_db_session,
            email_prefix="member",
            tenant=tenant,
            role=TenantAccountRole.NORMAL,
        )
        token = factory.create_owner_transfer_token(current_user)

        with (
            patch.object(members_module.AccountService, "send_new_owner_transfer_notify_email") as new_owner_email,
            patch.object(members_module.AccountService, "send_old_owner_transfer_notify_email") as old_owner_email,
        ):
            response = test_client_with_containers.post(
                f"/console/api/workspaces/current/members/{member.id}/owner-transfer",
                headers=_headers(test_client_with_containers, current_user),
                json={"token": token},
            )

        assert response.status_code == 200
        assert response.get_json()["result"] == "success"
        assert factory.get_join(transactional_db_session, tenant=tenant, account=member).role == TenantAccountRole.OWNER
        assert (
            factory.get_join(transactional_db_session, tenant=tenant, account=current_user).role
            == TenantAccountRole.ADMIN
        )
        new_owner_email.assert_called_once()
        old_owner_email.assert_called_once()
