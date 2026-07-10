"""DB-backed integration tests for service API dataset segment endpoints."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest
from flask.testing import FlaskClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.ext_redis import redis_client
from models.dataset import ChildChunk, Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import (
    ApiTokenType,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SegmentType,
    SummaryStatus,
)
from models.model import ApiToken
from tests.test_containers_integration_tests.controllers.console.helpers import create_console_account_and_tenant

pytestmark = pytest.mark.requires_redis


def _create_dataset_graph(db_session: Session) -> tuple[Dataset, Document, DocumentSegment]:
    account, tenant = create_console_account_and_tenant(db_session)
    dataset = Dataset(
        tenant_id=tenant.id,
        name=f"Segment Dataset {uuid4()}",
        description="Segment integration dataset",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        created_by=account.id,
        permission="only_me",
        provider="vendor",
        enable_api=True,
    )
    db_session.add(dataset)
    db_session.commit()

    document = Document(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch=f"batch-{uuid4()}",
        name="segment-doc.txt",
        created_from=DocumentCreatedFrom.API,
        created_by=account.id,
        enabled=True,
        archived=False,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        word_count=4,
        tokens=5,
    )
    db_session.add(document)
    db_session.commit()

    segment = DocumentSegment(
        tenant_id=tenant.id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="Segment content for integration",
        word_count=4,
        tokens=5,
        keywords=["segment", "integration"],
        status=SegmentStatus.COMPLETED,
        created_by=account.id,
    )
    db_session.add(segment)
    db_session.commit()

    summary = DocumentSegmentSummary(
        dataset_id=dataset.id,
        document_id=document.id,
        chunk_id=segment.id,
        summary_content="DB summary",
        status=SummaryStatus.COMPLETED,
    )
    db_session.add(summary)

    api_token = ApiToken(
        tenant_id=tenant.id,
        type=ApiTokenType.DATASET,
        token=f"dataset-{uuid4().hex}",
    )
    db_session.add(api_token)
    db_session.commit()
    return dataset, document, segment


def _auth_headers(db_session: Session, dataset: Dataset) -> dict[str, str]:
    token = db_session.query(ApiToken).filter_by(tenant_id=dataset.tenant_id, type=ApiTokenType.DATASET).one()
    return {"Authorization": f"Bearer {token.token}"}


def test_list_segments_uses_real_services_and_service_api_shape(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, segment = _create_dataset_graph(transactional_db_session)
    segment_id = segment.id

    response = test_client_with_containers.get(
        f"/v1/datasets/{dataset.id}/documents/{document.id}/segments"
        "?page=1&limit=20&status=completed&keyword=integration",
        headers=_auth_headers(transactional_db_session, dataset),
    )

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"data", "doc_form", "total", "has_more", "limit", "page"}
    assert body["doc_form"] == "text_model"
    assert body["total"] == 1
    assert "total_pages" not in body
    assert body["data"][0]["id"] == segment_id
    assert body["data"][0]["summary"] == "DB summary"
    assert body["data"][0]["attachments"] == []
    assert body["data"][0]["child_chunks"] == []


def test_list_child_chunks_uses_real_segment_service(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, segment = _create_dataset_graph(transactional_db_session)
    child_chunk = ChildChunk(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        position=1,
        content="Child integration content",
        word_count=3,
        type=SegmentType.CUSTOMIZED,
        created_by=document.created_by,
    )
    transactional_db_session.add(child_chunk)
    transactional_db_session.commit()

    response = test_client_with_containers.get(
        f"/v1/datasets/{dataset.id}/documents/{document.id}/segments/{segment.id}/child_chunks"
        "?page=1&limit=20&keyword=integration",
        headers=_auth_headers(transactional_db_session, dataset),
    )

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"data", "total", "total_pages", "page", "limit"}
    assert body["total"] == 1
    assert body["data"][0]["content"] == "Child integration content"


def test_create_get_update_and_delete_segment_persist(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, _segment = _create_dataset_graph(transactional_db_session)
    dataset_id = dataset.id
    document_id = document.id
    headers = _auth_headers(transactional_db_session, dataset)
    segments_url = f"/v1/datasets/{dataset_id}/documents/{document_id}/segments"

    with patch("services.dataset_service.VectorService.create_segments_vector"):
        create_response = test_client_with_containers.post(
            segments_url,
            headers=headers,
            json={
                "segments": [
                    {
                        "content": "Service API created segment",
                        "keywords": ["service-api"],
                        "attachment_ids": [],
                    }
                ]
            },
        )

    assert create_response.status_code == 200
    created_id = create_response.get_json()["data"][0]["id"]
    assert transactional_db_session.get(DocumentSegment, created_id).content == "Service API created segment"
    segment_url = f"{segments_url}/{created_id}"

    get_response = test_client_with_containers.get(segment_url, headers=headers)
    assert get_response.status_code == 200
    assert get_response.get_json()["data"]["id"] == created_id

    with (
        patch("services.dataset_service.VectorService.update_segment_vector"),
        patch("services.dataset_service.VectorService.update_multimodel_vector"),
    ):
        update_response = test_client_with_containers.post(
            segment_url,
            headers=headers,
            json={
                "segment": {
                    "content": "Service API updated segment",
                    "keywords": ["updated"],
                    "attachment_ids": [],
                }
            },
        )

    assert update_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, created_id).content == "Service API updated segment"

    with patch("services.dataset_service.delete_segment_from_index_task.delay") as delete_index:
        delete_response = test_client_with_containers.delete(segment_url, headers=headers)

    assert delete_response.status_code == 204
    assert transactional_db_session.get(DocumentSegment, created_id) is None
    delete_index.assert_called_once()


def test_create_update_and_delete_child_chunk_persist(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, segment = _create_dataset_graph(transactional_db_session)
    dataset_id = dataset.id
    document_id = document.id
    segment_id = segment.id
    headers = _auth_headers(transactional_db_session, dataset)
    child_chunks_url = (
        f"/v1/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}/child_chunks"
    )

    with patch("services.dataset_service.VectorService.create_child_chunk_vector"):
        create_response = test_client_with_containers.post(
            child_chunks_url,
            headers=headers,
            json={"content": "Service API child chunk"},
        )

    assert create_response.status_code == 200
    child_chunk_id = create_response.get_json()["data"]["id"]
    assert transactional_db_session.get(ChildChunk, child_chunk_id).content == "Service API child chunk"
    child_chunk_url = f"{child_chunks_url}/{child_chunk_id}"

    with patch("services.dataset_service.VectorService.update_child_chunk_vector"):
        update_response = test_client_with_containers.patch(
            child_chunk_url,
            headers=headers,
            json={"content": "Service API updated child"},
        )

    assert update_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(ChildChunk, child_chunk_id).content == "Service API updated child"

    with patch("services.dataset_service.VectorService.delete_child_chunk_vector"):
        delete_response = test_client_with_containers.delete(child_chunk_url, headers=headers)

    assert delete_response.status_code == 204
    remaining = transactional_db_session.scalar(
        select(func.count()).select_from(ChildChunk).where(ChildChunk.id == child_chunk_id)
    )
    assert remaining == 0


def test_update_keyword_status_and_index_failure_contracts(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, segment = _create_dataset_graph(transactional_db_session)
    dataset_id = dataset.id
    document_id = document.id
    segment_id = segment.id
    original_content = segment.content
    headers = _auth_headers(transactional_db_session, dataset)
    segment_url = f"/v1/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"

    with (
        patch("services.dataset_service.VectorService.update_segment_vector") as update_vector,
        patch("services.dataset_service.VectorService.update_multimodel_vector"),
    ):
        keyword_response = test_client_with_containers.post(
            segment_url,
            headers=headers,
            json={"segment": {"content": original_content, "keywords": ["replacement"]}},
        )
    assert keyword_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, segment_id).keywords == ["replacement"]
    update_vector.assert_called_once()

    with patch("services.dataset_service.disable_segment_from_index_task.delay") as disable_index:
        disable_response = test_client_with_containers.post(
            segment_url,
            headers=headers,
            json={"segment": {"enabled": False}},
        )
    assert disable_response.status_code == 200
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, segment_id).enabled is False
    disable_index.assert_called_once_with(segment_id)

    blocked_enable_response = test_client_with_containers.post(
        segment_url,
        headers=headers,
        json={"segment": {"enabled": True, "content": "Blocked by indexing lock"}},
    )
    assert blocked_enable_response.status_code == 400
    transactional_db_session.expire_all()
    assert transactional_db_session.get(DocumentSegment, segment_id).enabled is False

    redis_client.delete(f"segment_{segment_id}_indexing")

    with (
        patch("services.dataset_service.VectorService.update_segment_vector"),
        patch("services.dataset_service.VectorService.update_multimodel_vector"),
    ):
        enable_response = test_client_with_containers.post(
            segment_url,
            headers=headers,
            json={"segment": {"enabled": True, "content": "Re-enabled content"}},
        )
    assert enable_response.status_code == 200
    transactional_db_session.expire_all()
    reenabled_segment = transactional_db_session.get(DocumentSegment, segment_id)
    assert reenabled_segment.enabled is True
    assert reenabled_segment.content == "Re-enabled content"

    with patch(
        "services.dataset_service.VectorService.update_segment_vector",
        side_effect=RuntimeError("update index unavailable"),
    ):
        failed_update_response = test_client_with_containers.post(
            segment_url,
            headers=headers,
            json={"segment": {"content": "Persisted before index failure"}},
        )
    assert failed_update_response.status_code == 200
    transactional_db_session.expire_all()
    failed_segment = transactional_db_session.get(DocumentSegment, segment_id)
    assert failed_segment.content == "Persisted before index failure"
    assert failed_segment.status == SegmentStatus.ERROR
    assert failed_segment.enabled is False
    assert failed_segment.error == "update index unavailable"


def test_create_preconditions_qa_and_batch_index_failure_contracts(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, _segment = _create_dataset_graph(transactional_db_session)
    dataset_id = dataset.id
    document_id = document.id
    headers = _auth_headers(transactional_db_session, dataset)
    segments_url = f"/v1/datasets/{dataset_id}/documents/{document_id}/segments"

    document.indexing_status = IndexingStatus.INDEXING
    transactional_db_session.commit()
    incomplete_response = test_client_with_containers.post(
        segments_url,
        headers=headers,
        json={"segments": [{"content": "Rejected while indexing"}]},
    )
    assert incomplete_response.status_code == 404

    persisted_document = transactional_db_session.get(Document, document_id)
    persisted_document.indexing_status = IndexingStatus.COMPLETED
    persisted_document.enabled = False
    transactional_db_session.commit()
    disabled_response = test_client_with_containers.post(
        segments_url,
        headers=headers,
        json={"segments": [{"content": "Rejected while disabled"}]},
    )
    assert disabled_response.status_code == 404

    persisted_document = transactional_db_session.get(Document, document_id)
    persisted_document.enabled = True
    persisted_document.doc_form = IndexStructureType.QA_INDEX
    transactional_db_session.commit()
    with patch("services.dataset_service.VectorService.create_segments_vector"):
        qa_response = test_client_with_containers.post(
            segments_url,
            headers=headers,
            json={"segments": [{"content": "QA question", "answer": "QA answer"}]},
        )
    assert qa_response.status_code == 200
    qa_segment_id = qa_response.get_json()["data"][0]["id"]
    qa_segment = transactional_db_session.get(DocumentSegment, qa_segment_id)
    assert qa_segment.answer == "QA answer"
    assert qa_segment.word_count == len("QA question") + len("QA answer")

    with patch(
        "services.dataset_service.VectorService.create_segments_vector",
        side_effect=RuntimeError("batch index unavailable"),
    ):
        failed_batch_response = test_client_with_containers.post(
            segments_url,
            headers=headers,
            json={
                "segments": [
                    {"content": "Failed question one", "answer": "Answer one"},
                    {"content": "Failed question two", "answer": "Answer two"},
                ]
            },
        )
    assert failed_batch_response.status_code == 200
    failed_ids = [item["id"] for item in failed_batch_response.get_json()["data"]]
    failed_segments = transactional_db_session.scalars(
        select(DocumentSegment).where(DocumentSegment.id.in_(failed_ids))
    ).all()
    assert len(failed_segments) == 2
    assert all(segment.status == SegmentStatus.ERROR for segment in failed_segments)
    assert all(segment.enabled is False for segment in failed_segments)
    assert {segment.error for segment in failed_segments} == {"batch index unavailable"}


def test_resource_hierarchy_mismatches_are_not_found(
    test_client_with_containers: FlaskClient,
    transactional_db_session: Session,
) -> None:
    dataset, document, segment = _create_dataset_graph(transactional_db_session)
    dataset_id = dataset.id
    document_id = document.id
    segment_id = segment.id
    headers = _auth_headers(transactional_db_session, dataset)
    unknown_id = uuid4()

    responses = [
        test_client_with_containers.get(
            f"/v1/datasets/{unknown_id}/documents/{document_id}/segments",
            headers=headers,
        ),
        test_client_with_containers.get(
            f"/v1/datasets/{dataset_id}/documents/{unknown_id}/segments",
            headers=headers,
        ),
        test_client_with_containers.get(
            f"/v1/datasets/{dataset_id}/documents/{document_id}/segments/{unknown_id}",
            headers=headers,
        ),
        test_client_with_containers.patch(
            f"/v1/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"
            f"/child_chunks/{unknown_id}",
            headers=headers,
            json={"content": "Unknown child"},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
