from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import pytest
from flask.testing import FlaskClient
from sqlalchemy.orm import Session

from models import Account, App
from services.app_service import AppService, CreateAppParams
from tests.test_containers_integration_tests.controllers.openapi.conftest import BearerFactory

pytestmark = pytest.mark.requires_redis


def _create_app(
    db_session: Session,
    account: Account,
    *,
    name: str,
    enable_api: bool = True,
) -> App:
    tenant = account.current_tenant
    assert tenant is not None
    params = CreateAppParams(
        name=name,
        description="",
        mode="workflow",
        icon_type="emoji",
        icon="robot",
        icon_background="#FF6B6B",
    )
    app_model = AppService().create_app(tenant.id, params, account, session=db_session)
    # The openapi surface gate keys off ``enable_api``; flip it explicitly so
    # the test states the visibility precondition rather than relying on the
    # template default.
    app_model.enable_api = enable_api
    db_session.add(app_model)
    db_session.commit()
    return app_model


class TestAppList:
    def test_lists_only_api_enabled_apps_in_workspace(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        tenant = account.current_tenant
        assert tenant is not None
        visible = _create_app(transactional_db_session, account, name="Visible", enable_api=True)
        _create_app(transactional_db_session, account, name="Hidden", enable_api=False)
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(
            f"/openapi/v1/apps?workspace_id={tenant.id}", headers=headers
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["total"] == 1
        assert [row["id"] for row in result["data"]] == [visible.id]
        assert result["has_more"] is False

    def test_uuid_name_filter_returns_matching_app(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        tenant = account.current_tenant
        assert tenant is not None
        target = _create_app(transactional_db_session, account, name="Target", enable_api=True)
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(
            f"/openapi/v1/apps?workspace_id={tenant.id}&name={target.id}", headers=headers
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["total"] == 1
        assert result["data"][0]["id"] == target.id
        assert result["data"][0]["workspace_id"] == str(tenant.id)

    def test_uuid_name_filter_for_foreign_app_returns_empty(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        owner = make_transactional_account()
        outsider = make_transactional_account()
        foreign_app = _create_app(transactional_db_session, owner, name="Foreign", enable_api=True)
        outsider_tenant = outsider.current_tenant
        assert outsider_tenant is not None
        headers, _mint = account_bearer_factory(outsider)

        response = test_client_with_containers.get(
            f"/openapi/v1/apps?workspace_id={outsider_tenant.id}&name={foreign_app.id}", headers=headers
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["total"] == 0
        assert result["data"] == []


class TestAppDescribe:
    def test_describe_info_returns_metadata(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        app_model = _create_app(transactional_db_session, account, name="Describe Me", enable_api=True)
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(
            f"/openapi/v1/apps/{app_model.id}?fields=info", headers=headers
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["info"]["id"] == app_model.id
        assert result["info"]["name"] == "Describe Me"
        assert result["info"]["service_api_enabled"] is True
        assert result["parameters"] is None
        assert result["input_schema"] is None

    def test_describe_unknown_app_is_404(
        self,
        test_client_with_containers: FlaskClient,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        missing_id = str(uuid4())
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(f"/openapi/v1/apps/{missing_id}", headers=headers)

        assert response.status_code == 404
        assert response.get_json()["message"] == "app not found"

    def test_describe_api_disabled_app_is_403(
        self,
        test_client_with_containers: FlaskClient,
        transactional_db_session: Session,
        make_transactional_account: Callable[..., Account],
        account_bearer_factory: BearerFactory,
    ) -> None:
        account = make_transactional_account()
        hidden = _create_app(transactional_db_session, account, name="Hidden", enable_api=False)
        headers, _mint = account_bearer_factory(account)

        response = test_client_with_containers.get(f"/openapi/v1/apps/{hidden.id}", headers=headers)

        assert response.status_code == 403
        assert response.get_json()["message"] == "service_api_disabled"
