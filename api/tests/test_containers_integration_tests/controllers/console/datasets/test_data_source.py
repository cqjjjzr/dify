"""Testcontainers integration tests for controllers.console.datasets.data_source endpoints."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import uuid4

import pytest
from flask import Flask
from sqlalchemy.orm import Session
from werkzeug.exceptions import NotFound

from controllers.console.datasets import data_source
from controllers.console.datasets.data_source import (
    DataSourceNotionDatasetSyncApi,
    DataSourceNotionDocumentSyncApi,
    DataSourceNotionIndexingEstimateApi,
    DataSourceNotionListApi,
    DataSourceNotionPreviewApi,
)
from core.rag.index_processor.constant.index_type import IndexStructureType
from models import Account, DataSourceOauthBinding
from models.dataset import Document
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus
from tests.test_containers_integration_tests.controllers.console.helpers import AuthenticatedConsoleClient
from tests.test_containers_integration_tests.transactional import DatabaseState


@pytest.fixture
def current_user() -> Account:
    account = Account(name="Test User", email="u1@example.com")
    account.id = "u1"
    return account


@pytest.fixture
def mock_engine() -> Iterator[None]:
    with patch.object(
        type(data_source.db),
        "engine",
        new_callable=PropertyMock,
        return_value=MagicMock(),
    ):
        yield


class TestDataSourceApi:
    @staticmethod
    def create_binding(
        session: Session,
        *,
        tenant_id: str,
        disabled: bool = False,
    ) -> DataSourceOauthBinding:
        binding = DataSourceOauthBinding(
            tenant_id=tenant_id,
            access_token="token",
            provider="notion",
            source_info={
                "workspace_name": "Workspace",
                "workspace_id": "workspace-1",
                "workspace_icon": None,
                "total": 1,
                "pages": [
                    {
                        "page_id": "page-1",
                        "page_name": "Page",
                        "page_icon": {"type": "emoji", "emoji": "P", "url": None},
                        "parent_id": "parent-1",
                        "type": "page",
                    }
                ],
            },
            disabled=disabled,
        )
        session.add(binding)
        session.commit()
        return binding

    def test_get_success(
        self,
        authenticated_console_client: AuthenticatedConsoleClient,
        transactional_db_session: Session,
    ) -> None:
        binding = self.create_binding(
            transactional_db_session,
            tenant_id=authenticated_console_client.tenant.id,
        )
        binding_id = binding.id
        binding_created_at = int(binding.created_at.timestamp())

        response = authenticated_console_client.client.get(
            "/console/api/data-source/integrates",
            headers=authenticated_console_client.headers,
        )

        assert response.status_code == 200
        assert response.json is not None
        assert response.json["data"][0] == {
            "id": binding_id,
            "provider": "notion",
            "created_at": binding_created_at,
            "is_bound": True,
            "disabled": False,
            "source_info": {
                "workspace_name": "Workspace",
                "workspace_id": "workspace-1",
                "workspace_icon": None,
                "pages": [
                    {
                        "page_name": "Page",
                        "page_id": "page-1",
                        "page_icon": {"type": "emoji", "url": None, "emoji": "P"},
                        "parent_id": "parent-1",
                        "type": "page",
                    }
                ],
                "total": 1,
            },
            "link": "http://localhost/console/api/oauth/data-source/notion",
        }

    def test_get_no_bindings(self, authenticated_console_client: AuthenticatedConsoleClient) -> None:
        response = authenticated_console_client.client.get(
            "/console/api/data-source/integrates",
            headers=authenticated_console_client.headers,
        )

        assert response.status_code == 200
        assert response.json == {"data": []}

    @pytest.mark.parametrize(("initially_disabled", "action"), [(True, "enable"), (False, "disable")])
    def test_patch_binding_persists_state_change(
        self,
        initially_disabled: bool,
        action: str,
        authenticated_console_client: AuthenticatedConsoleClient,
        transactional_db_session: Session,
        database_state: DatabaseState,
    ) -> None:
        binding = self.create_binding(
            transactional_db_session,
            tenant_id=authenticated_console_client.tenant.id,
            disabled=initially_disabled,
        )
        binding_id = binding.id

        response = authenticated_console_client.client.patch(
            f"/console/api/data-source/integrates/{binding_id}/{action}",
            headers=authenticated_console_client.headers,
        )

        assert response.status_code == 200
        assert response.json == {"result": "success"}
        persisted = database_state.one(DataSourceOauthBinding, DataSourceOauthBinding.id == binding_id)
        assert persisted.disabled is (not initially_disabled)

    def test_patch_binding_not_found(self, authenticated_console_client: AuthenticatedConsoleClient) -> None:
        response = authenticated_console_client.client.patch(
            f"/console/api/data-source/integrates/{uuid4()}/enable",
            headers=authenticated_console_client.headers,
        )

        assert response.status_code == 404

    @pytest.mark.parametrize(("disabled", "action"), [(False, "enable"), (True, "disable")])
    def test_patch_binding_rejects_noop_without_changing_state(
        self,
        disabled: bool,
        action: str,
        authenticated_console_client: AuthenticatedConsoleClient,
        transactional_db_session: Session,
        database_state: DatabaseState,
    ) -> None:
        binding = self.create_binding(
            transactional_db_session,
            tenant_id=authenticated_console_client.tenant.id,
            disabled=disabled,
        )
        binding_id = binding.id

        response = authenticated_console_client.client.patch(
            f"/console/api/data-source/integrates/{binding_id}/{action}",
            headers=authenticated_console_client.headers,
        )

        assert response.status_code == 400
        persisted = database_state.one(DataSourceOauthBinding, DataSourceOauthBinding.id == binding_id)
        assert persisted.disabled is disabled


class TestDataSourceNotionListApi:
    @pytest.fixture
    def app(self, flask_app_with_containers: Flask) -> Flask:
        return flask_app_with_containers

    def test_get_credential_not_found(self, app: Flask, current_user: Account) -> None:
        api = DataSourceNotionListApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/?credential_id=c1"),
            patch(
                "controllers.console.datasets.data_source.DatasourceProviderService.get_datasource_credentials",
                return_value=None,
            ),
        ):
            with pytest.raises(NotFound):
                method(api, MagicMock(), "tenant-1", current_user)

    def test_get_success_no_dataset_id(self, app: Flask, current_user: Account, mock_engine: None) -> None:
        api = DataSourceNotionListApi()
        method = inspect.unwrap(api.get)

        page = MagicMock(
            page_id="p1",
            page_name="Page 1",
            type="page",
            parent_id="parent",
            page_icon=None,
        )

        online_document_message = MagicMock(
            result=[
                MagicMock(
                    workspace_id="w1",
                    workspace_name="My Workspace",
                    workspace_icon="icon",
                    pages=[page],
                )
            ]
        )

        with (
            app.test_request_context("/?credential_id=c1"),
            patch(
                "controllers.console.datasets.data_source.DatasourceProviderService.get_datasource_credentials",
                return_value={"token": "t"},
            ),
            patch(
                "core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime",
                return_value=MagicMock(
                    get_online_document_pages=lambda **kw: iter([online_document_message]),
                    datasource_provider_type=lambda: None,
                ),
            ),
        ):
            response, status = method(api, MagicMock(), "tenant-1", current_user)

        assert status == 200

    def test_get_success_with_dataset_id(
        self, app: Flask, current_user: Account, mock_engine: None, db_session_with_containers: Session
    ) -> None:
        api = DataSourceNotionListApi()
        method = inspect.unwrap(api.get)
        tenant_id = str(uuid4())
        dataset_id = str(uuid4())

        page = MagicMock(
            page_id="p1",
            page_name="Page 1",
            type="page",
            parent_id="parent",
            page_icon=None,
        )

        online_document_message = MagicMock(
            result=[
                MagicMock(
                    workspace_id="w1",
                    workspace_name="My Workspace",
                    workspace_icon="icon",
                    pages=[page],
                )
            ]
        )

        dataset = MagicMock(data_source_type="notion_import")
        document = Document(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            position=1,
            data_source_type=DataSourceType.NOTION_IMPORT,
            data_source_info='{"notion_page_id": "p1"}',
            batch=f"batch-{uuid4()}",
            name="Notion Page",
            created_from=DocumentCreatedFrom.WEB,
            created_by=str(uuid4()),
            indexing_status=IndexingStatus.COMPLETED,
            enabled=True,
        )
        db_session_with_containers.add(document)
        db_session_with_containers.commit()

        with (
            app.test_request_context(f"/?credential_id=c1&dataset_id={dataset_id}"),
            patch(
                "controllers.console.datasets.data_source.DatasourceProviderService.get_datasource_credentials",
                return_value={"token": "t"},
            ),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=dataset,
            ),
            patch(
                "core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime",
                return_value=MagicMock(
                    get_online_document_pages=lambda **kw: iter([online_document_message]),
                    datasource_provider_type=lambda: None,
                ),
            ),
        ):
            response, status = method(api, db_session_with_containers, tenant_id, current_user)

        assert status == 200

    def test_get_invalid_dataset_type(self, app: Flask, current_user: Account) -> None:
        api = DataSourceNotionListApi()
        method = inspect.unwrap(api.get)

        dataset = MagicMock(data_source_type="other_type")

        with (
            app.test_request_context("/?credential_id=c1&dataset_id=ds1"),
            patch(
                "controllers.console.datasets.data_source.DatasourceProviderService.get_datasource_credentials",
                return_value={"token": "t"},
            ),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=dataset,
            ),
        ):
            with pytest.raises(ValueError):
                method(api, MagicMock(), "tenant-1", current_user)


class TestDataSourceNotionPreviewApi:
    @pytest.fixture
    def app(self, flask_app_with_containers: Flask) -> Flask:
        return flask_app_with_containers

    def test_get_preview_success(self, app: Flask) -> None:
        api = DataSourceNotionPreviewApi()
        method = inspect.unwrap(api.get)

        extractor = MagicMock(extract=lambda: [MagicMock(page_content="hello")])

        with (
            app.test_request_context("/?credential_id=c1"),
            patch(
                "controllers.console.datasets.data_source.DatasourceProviderService.get_datasource_credentials",
                return_value={"integration_secret": "t"},
            ),
            patch(
                "controllers.console.datasets.data_source.NotionExtractor",
                return_value=extractor,
            ),
        ):
            response, status = method(api, "tenant-1", "p1", "page")

        assert status == 200


class TestDataSourceNotionIndexingEstimateApi:
    @pytest.fixture
    def app(self, flask_app_with_containers: Flask) -> Flask:
        return flask_app_with_containers

    def test_post_indexing_estimate_success(self, app: Flask) -> None:
        api = DataSourceNotionIndexingEstimateApi()
        method = inspect.unwrap(api.post)

        empty_rules: dict[str, object] = {}
        payload: dict[str, object] = {
            "notion_info_list": [
                {
                    "workspace_id": "w1",
                    "credential_id": "c1",
                    "pages": [{"page_id": "p1", "type": "page"}],
                }
            ],
            "process_rule": {"rules": empty_rules},
            "doc_form": IndexStructureType.PARAGRAPH_INDEX,
            "doc_language": "English",
        }

        with (
            app.test_request_context("/", method="POST", json=payload, headers={"Content-Type": "application/json"}),
            patch(
                "controllers.console.datasets.data_source.DocumentService.estimate_args_validate",
            ),
            patch(
                "controllers.console.datasets.data_source.IndexingRunner.indexing_estimate",
                return_value=MagicMock(model_dump=lambda: {"total_pages": 1}),
            ),
        ):
            response, status = method(api, MagicMock(), "tenant-1")

        assert status == 200


class TestDataSourceNotionDatasetSyncApi:
    @pytest.fixture
    def app(self, flask_app_with_containers: Flask) -> Flask:
        return flask_app_with_containers

    def test_get_success(self, app: Flask) -> None:
        api = DataSourceNotionDatasetSyncApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/"),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=MagicMock(),
            ),
            patch(
                "controllers.console.datasets.data_source.DocumentService.get_document_by_dataset_id",
                return_value=[MagicMock(id="d1")],
            ),
            patch(
                "controllers.console.datasets.data_source.document_indexing_sync_task.delay",
                return_value=None,
            ),
        ):
            response, status = method(api, MagicMock(), "ds-1")

        assert status == 200

    def test_get_dataset_not_found(self, app: Flask) -> None:
        api = DataSourceNotionDatasetSyncApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/"),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=None,
            ),
        ):
            with pytest.raises(NotFound):
                method(api, MagicMock(), "ds-1")


class TestDataSourceNotionDocumentSyncApi:
    @pytest.fixture
    def app(self, flask_app_with_containers: Flask) -> Flask:
        return flask_app_with_containers

    def test_get_success(self, app: Flask) -> None:
        api = DataSourceNotionDocumentSyncApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/"),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=MagicMock(),
            ),
            patch(
                "controllers.console.datasets.data_source.DocumentService.get_document",
                return_value=MagicMock(),
            ),
            patch(
                "controllers.console.datasets.data_source.document_indexing_sync_task.delay",
                return_value=None,
            ),
        ):
            response, status = method(api, MagicMock(), "ds-1", "doc-1")

        assert status == 200

    def test_get_document_not_found(self, app: Flask) -> None:
        api = DataSourceNotionDocumentSyncApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/"),
            patch(
                "controllers.console.datasets.data_source.DatasetService.get_dataset",
                return_value=MagicMock(),
            ),
            patch(
                "controllers.console.datasets.data_source.DocumentService.get_document",
                return_value=None,
            ),
        ):
            with pytest.raises(NotFound):
                method(api, MagicMock(), "ds-1", "doc-1")
