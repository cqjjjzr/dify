from __future__ import annotations

import pytest
from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from models import Account, Tenant
from models.account import TenantAccountRole
from tests.test_containers_integration_tests.controllers.console.helpers import (
    AuthenticatedConsoleClient,
    ConsoleAccountFactory,
    authenticate_console_client,
    create_console_account_and_tenant,
)


@pytest.fixture
def console_account_factory(transactional_db_session: Session) -> ConsoleAccountFactory:
    def create(*, role: TenantAccountRole = TenantAccountRole.OWNER) -> tuple[Account, Tenant]:
        return create_console_account_and_tenant(transactional_db_session, role=role)

    return create


@pytest.fixture
def authenticated_console_client(
    console_account_factory: ConsoleAccountFactory,
    test_client_with_containers: FlaskClient,
) -> AuthenticatedConsoleClient:
    account, tenant = console_account_factory()
    return AuthenticatedConsoleClient(
        client=test_client_with_containers,
        headers=authenticate_console_client(test_client_with_containers, account),
        account=account,
        tenant=tenant,
    )
