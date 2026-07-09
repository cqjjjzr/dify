from __future__ import annotations

from collections.abc import Callable

import pytest
from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from extensions.ext_redis import redis_client
from models import Account, OAuthAccessToken
from services.oauth_device_flow import PREFIX_OAUTH_ACCOUNT, MintResult, mint_oauth_token
from tests.test_containers_integration_tests.controllers.openapi.conftest import BearerFactory
from tests.test_containers_integration_tests.transactional import DatabaseState

pytestmark = pytest.mark.requires_redis


def _mint_account_token(
    db_session: Session,
    account: Account,
    *,
    client_id: str = "integration-cli",
    device_label: str = "Test Device",
) -> MintResult:
    return mint_oauth_token(
        redis_client,
        subject_email=account.email,
        subject_issuer=None,
        account_id=str(account.id),
        client_id=client_id,
        device_label=device_label,
        prefix=PREFIX_OAUTH_ACCOUNT,
        ttl_days=14,
        session=db_session,
    )


class TestSessionList:
    def test_lists_active_session(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
    ) -> None:
        account = make_transactional_account()
        mint = _mint_account_token(transactional_db_session, account, device_label="Laptop")

        response = test_client_with_containers.get(
            "/openapi/v1/account/sessions", headers={"Authorization": f"Bearer {mint.token}"}
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["total"] == 1
        row = result["data"][0]
        assert row["id"] == str(mint.token_id)
        assert row["prefix"] == PREFIX_OAUTH_ACCOUNT
        assert row["device_label"] == "Laptop"

    def test_excludes_other_accounts_sessions(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
    ) -> None:
        account = make_transactional_account()
        other = make_transactional_account()
        mine = _mint_account_token(transactional_db_session, account)
        _mint_account_token(transactional_db_session, other)

        response = test_client_with_containers.get(
            "/openapi/v1/account/sessions", headers={"Authorization": f"Bearer {mine.token}"}
        )

        assert response.status_code == 200
        result = response.get_json()
        assert {row["id"] for row in result["data"]} == {str(mine.token_id)}


class TestSessionRevoke:
    def test_revoke_self_persists(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        database_state: DatabaseState,
    ) -> None:
        account = make_transactional_account()
        mint = _mint_account_token(transactional_db_session, account)

        response = test_client_with_containers.delete(
            "/openapi/v1/account/sessions/self", headers={"Authorization": f"Bearer {mint.token}"}
        )

        assert response.status_code == 200
        assert response.get_json() == {"status": "revoked"}
        persisted = database_state.one(OAuthAccessToken, OAuthAccessToken.id == mint.token_id)
        assert persisted.revoked_at is not None

    def test_revoke_by_id_for_own_session(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
        database_state: DatabaseState,
    ) -> None:
        account = make_transactional_account()
        caller_headers, _caller = account_bearer_factory(account)
        target = _mint_account_token(transactional_db_session, account, client_id="target-client")

        response = test_client_with_containers.delete(
            f"/openapi/v1/account/sessions/{target.token_id}", headers=caller_headers
        )

        assert response.status_code == 200
        assert response.get_json() == {"status": "revoked"}
        persisted = database_state.one(OAuthAccessToken, OAuthAccessToken.id == target.token_id)
        assert persisted.revoked_at is not None

    def test_revoke_foreign_session_is_404(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
        database_state: DatabaseState,
    ) -> None:
        owner = make_transactional_account()
        outsider = make_transactional_account()
        foreign = _mint_account_token(transactional_db_session, owner)
        outsider_headers, _outsider = account_bearer_factory(outsider)

        response = test_client_with_containers.delete(
            f"/openapi/v1/account/sessions/{foreign.token_id}", headers=outsider_headers
        )

        assert response.status_code == 404
        assert response.get_json()["message"] == "session not found"
        persisted = database_state.one(OAuthAccessToken, OAuthAccessToken.id == foreign.token_id)
        assert persisted.revoked_at is None
