"""DB-backed integration tests for console dataset segment endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from flask.testing import FlaskClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.storage.storage_type import StorageType
from models.dataset import ChildChunk, Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import (
    CreatorUserRole,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SegmentType,
    SummaryStatus,
)
from models.model import UploadFile
from tests.test_containers_integration_tests.controllers.console.helpers import (
    authenticate_console_client,
    create_console_account_and_tenant,
)

pytestmark = pytest.mark.requires_redis


@dataclass(frozen=True)
class ConsoleSegmentGraph:
    tenant_id: str
    account_id: str
    dataset_id: str
    document_id: str
    segment_id: str
    child_chunk_id: str
    headers: dict[str, str]


@pytest.fixture
def console_segment_graph(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> ConsoleSegmentGraph:
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    dataset = Dataset(
        tenant_id=tenant.id,
        name=f"Mutable Console Segment Dataset {uuid4()}",
        description="Console segment mutation dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        created_by=account.id,
        permission="only_me",
        provider="vendor",
    )
    transactional_db_session.add(dataset)
    transactional_db_session.commit()
    document = Document(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch=f"batch-{uuid4()}",
        name="mutable-segments.txt",
        created_from=DocumentCreatedFrom.WEB,
        created_by=account.id,
        enabled=True,
        archived=False,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARENT_CHILD_INDEX,
        word_count=24,
        tokens=5,
    )
    transactional_db_session.add(document)
    transactional_db_session.commit()
    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        index_node_id=str(uuid4()),
        position=1,
        content="Mutable parent segment",
        word_count=24,
        tokens=5,
        keywords=["mutable"],
        status=SegmentStatus.COMPLETED,
        created_by=account.id,
    )
    transactional_db_session.add(segment)
    transactional_db_session.commit()
    child_chunk = ChildChunk(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        index_node_id=str(uuid4()),
        position=1,
        content="Initial child chunk",
        word_count=3,
        type=SegmentType.CUSTOMIZED,
        created_by=account.id,
    )
    transactional_db_session.add(child_chunk)
    transactional_db_session.commit()
    return ConsoleSegmentGraph(
        tenant_id=tenant.id,
        account_id=account.id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        child_chunk_id=child_chunk.id,
        headers=authenticate_console_client(test_client_with_containers, account),
    )


def _segments_url(graph: ConsoleSegmentGraph) -> str:
    return f"/console/api/datasets/{graph.dataset_id}/documents/{graph.document_id}/segments"


def test_list_segments_uses_real_db_query_and_console_response_shape(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    account, tenant = create_console_account_and_tenant(transactional_db_session)
    dataset = Dataset(
        tenant_id=tenant.id,
        name=f"Console Segment Dataset {uuid4()}",
        description="Console segment integration dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        created_by=account.id,
        permission="only_me",
        provider="vendor",
    )
    transactional_db_session.add(dataset)
    transactional_db_session.commit()

    document = Document(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch=f"batch-{uuid4()}",
        name="console-segment-doc.txt",
        created_from=DocumentCreatedFrom.WEB,
        created_by=account.id,
        enabled=True,
        archived=False,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        word_count=3,
        tokens=4,
    )
    transactional_db_session.add(document)
    transactional_db_session.commit()

    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="Console integration segment",
        word_count=3,
        tokens=4,
        keywords=["console", "integration"],
        status=SegmentStatus.COMPLETED,
        created_by=account.id,
    )
    transactional_db_session.add(segment)
    transactional_db_session.commit()
    segment_id = segment.id

    transactional_db_session.add(
        DocumentSegmentSummary(
            dataset_id=dataset.id,
            document_id=document.id,
            chunk_id=segment.id,
            summary_content="Console DB summary",
            status=SummaryStatus.COMPLETED,
        )
    )
    transactional_db_session.commit()

    response = test_client_with_containers.get(
        f"/console/api/datasets/{dataset.id}/documents/{document.id}/segments"
        "?page=1&limit=10&status=completed&keyword=integration&enabled=all",
        headers=authenticate_console_client(test_client_with_containers, account),
    )

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"data", "limit", "total", "total_pages", "page"}
    assert body["limit"] == 10
    assert body["total"] == 1
    assert body["total_pages"] == 1
    assert "has_more" not in body
    assert body["data"][0]["id"] == segment_id
    assert body["data"][0]["summary"] == "Console DB summary"


def test_bulk_delete_segments_persists(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    with patch("services.dataset_service.delete_segment_from_index_task.delay") as delete_index:
        response = test_client_with_containers.delete(
            f"{_segments_url(graph)}?segment_id={graph.segment_id}",
            headers=graph.headers,
        )

    assert response.status_code == 204
    assert (
        transactional_db_session.scalar(
            select(func.count()).select_from(DocumentSegment).where(DocumentSegment.id == graph.segment_id)
        )
        == 0
    )
    delete_index.assert_called_once()


def test_disable_and_enable_segment_persist_status(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    with patch("services.dataset_service.disable_segments_from_index_task.delay") as disable_index:
        response = test_client_with_containers.patch(
            f"/console/api/datasets/{graph.dataset_id}/documents/{graph.document_id}/segment/disable"
            f"?segment_id={graph.segment_id}",
            headers=graph.headers,
        )

    assert response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, graph.segment_id).enabled is False
    disable_index.assert_called_once()

    with patch("services.dataset_service.enable_segments_to_index_task.delay") as enable_index:
        enable_response = test_client_with_containers.patch(
            f"/console/api/datasets/{graph.dataset_id}/documents/{graph.document_id}/segment/enable"
            f"?segment_id={graph.segment_id}",
            headers=graph.headers,
        )

    assert enable_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, graph.segment_id).enabled is True
    enable_index.assert_called_once_with([graph.segment_id], graph.dataset_id, graph.document_id)


def test_create_update_and_delete_segment_persist(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    with patch("services.dataset_service.VectorService.create_segments_vector"):
        create_response = test_client_with_containers.post(
            f"/console/api/datasets/{graph.dataset_id}/documents/{graph.document_id}/segment",
            headers=graph.headers,
            json={"content": "Created through HTTP", "keywords": ["created"], "attachment_ids": []},
        )

    assert create_response.status_code == 200
    created_id = create_response.get_json()["data"]["id"]
    assert transactional_db_session.get(DocumentSegment, created_id).content == "Created through HTTP"

    with patch("services.dataset_service.VectorService.update_multimodel_vector"):
        update_response = test_client_with_containers.patch(
            f"{_segments_url(graph)}/{created_id}",
            headers=graph.headers,
            json={
                "content": "Updated through HTTP",
                "keywords": ["updated"],
                "attachment_ids": [],
            },
        )

    assert update_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, created_id).content == "Updated through HTTP"

    with patch("services.dataset_service.delete_segment_from_index_task.delay") as delete_index:
        delete_response = test_client_with_containers.delete(
            f"{_segments_url(graph)}/{created_id}",
            headers=graph.headers,
        )

    assert delete_response.status_code == 204
    assert transactional_db_session.get(DocumentSegment, created_id) is None
    delete_index.assert_called_once()


def test_segment_index_failure_persists_error_state(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    with patch(
        "services.dataset_service.VectorService.create_segments_vector",
        side_effect=RuntimeError("index unavailable"),
    ):
        response = test_client_with_containers.post(
            f"/console/api/datasets/{graph.dataset_id}/documents/{graph.document_id}/segment",
            headers=graph.headers,
            json={"content": "Persisted index failure", "attachment_ids": []},
        )

    assert response.status_code == 200
    segment = transactional_db_session.scalars(
        select(DocumentSegment).where(DocumentSegment.content == "Persisted index failure")
    ).one()
    assert segment.status == SegmentStatus.ERROR
    assert segment.enabled is False
    assert segment.error == "index unavailable"


def test_batch_import_start_and_status_use_real_redis_state(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    upload_file = UploadFile(
        tenant_id=graph.tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"segment-import/{uuid4()}.csv",
        name="segments.csv",
        size=128,
        extension=".csv",
        mime_type="text/csv",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=graph.account_id,
        created_at=datetime.now(UTC),
        used=False,
    )
    transactional_db_session.add(upload_file)
    transactional_db_session.commit()
    upload_file_id = upload_file.id

    with patch(
        "controllers.console.datasets.datasets_segments.batch_create_segment_to_index_task.delay"
    ) as batch_task:
        start_response = test_client_with_containers.post(
            f"{_segments_url(graph)}/batch_import",
            headers=graph.headers,
            json={"upload_file_id": upload_file_id},
        )

    assert start_response.status_code == 200
    job_id = start_response.get_json()["job_id"]
    assert start_response.get_json()["job_status"] == "waiting"
    batch_task.assert_called_once()

    status_response = test_client_with_containers.get(
        f"/console/api/datasets/batch_import_status/{job_id}",
        headers=graph.headers,
    )
    assert status_response.status_code == 200
    assert status_response.get_json() == {"job_id": job_id, "job_status": "waiting"}


def test_create_and_list_child_chunks_persist(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    child_url = f"{_segments_url(graph)}/{graph.segment_id}/child_chunks"
    with patch("services.dataset_service.VectorService.create_child_chunk_vector"):
        create_response = test_client_with_containers.post(
            child_url,
            headers=graph.headers,
            json={"content": "Created child through HTTP"},
        )

    assert create_response.status_code == 200
    created_id = create_response.get_json()["data"]["id"]
    assert transactional_db_session.get(ChildChunk, created_id).content == "Created child through HTTP"

    list_response = test_client_with_containers.get(child_url, headers=graph.headers)
    assert list_response.status_code == 200
    assert {item["id"] for item in list_response.get_json()["data"]} == {
        graph.child_chunk_id,
        created_id,
    }


def test_batch_update_child_chunks_persists(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    child_url = f"{_segments_url(graph)}/{graph.segment_id}/child_chunks"
    with patch("services.dataset_service.VectorService.update_child_chunk_vector"):
        response = test_client_with_containers.patch(
            child_url,
            headers=graph.headers,
            json={"chunks": [{"id": graph.child_chunk_id, "content": "Batch updated child"}]},
        )

    assert response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(ChildChunk, graph.child_chunk_id).content == "Batch updated child"


def test_update_and_delete_child_chunk_persist(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    child_url = f"{_segments_url(graph)}/{graph.segment_id}/child_chunks/{graph.child_chunk_id}"
    with patch("services.dataset_service.VectorService.update_child_chunk_vector"):
        update_response = test_client_with_containers.patch(
            child_url,
            headers=graph.headers,
            json={"content": "Individually updated child"},
        )

    assert update_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(ChildChunk, graph.child_chunk_id).content == "Individually updated child"

    with patch("services.dataset_service.VectorService.delete_child_chunk_vector"):
        delete_response = test_client_with_containers.delete(child_url, headers=graph.headers)

    assert delete_response.status_code == 204
    assert transactional_db_session.get(ChildChunk, graph.child_chunk_id) is None


def test_child_chunk_index_failures_roll_back_all_mutations(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
    console_segment_graph: ConsoleSegmentGraph,
) -> None:
    graph = console_segment_graph
    child_chunks_url = f"{_segments_url(graph)}/{graph.segment_id}/child_chunks"
    original_count = transactional_db_session.scalar(
        select(func.count()).select_from(ChildChunk).where(ChildChunk.segment_id == graph.segment_id)
    )

    with patch(
        "services.dataset_service.VectorService.create_child_chunk_vector",
        side_effect=RuntimeError("create index failed"),
    ):
        create_response = test_client_with_containers.post(
            child_chunks_url,
            headers=graph.headers,
            json={"content": "Must roll back"},
        )

    assert create_response.status_code == 500
    assert (
        transactional_db_session.scalar(
            select(func.count()).select_from(ChildChunk).where(ChildChunk.segment_id == graph.segment_id)
        )
        == original_count
    )

    child_chunk_url = f"{child_chunks_url}/{graph.child_chunk_id}"
    with patch(
        "services.dataset_service.VectorService.update_child_chunk_vector",
        side_effect=RuntimeError("update index failed"),
    ):
        update_response = test_client_with_containers.patch(
            child_chunk_url,
            headers=graph.headers,
            json={"content": "Must not persist"},
        )

    assert update_response.status_code == 500
    transactional_db_session.expire_all()
    assert transactional_db_session.get(ChildChunk, graph.child_chunk_id).content == "Initial child chunk"

    with patch(
        "services.dataset_service.VectorService.delete_child_chunk_vector",
        side_effect=RuntimeError("delete index failed"),
    ):
        delete_response = test_client_with_containers.delete(child_chunk_url, headers=graph.headers)

    assert delete_response.status_code == 500
    transactional_db_session.expire_all()
    assert transactional_db_session.get(ChildChunk, graph.child_chunk_id) is not None
