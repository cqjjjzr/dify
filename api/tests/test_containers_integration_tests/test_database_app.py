from flask import Flask
from sqlalchemy import text
from sqlalchemy.orm import Session


def test_database_app_initializes_only_database_dependencies(
    database_app_with_containers: Flask,
    database_session_with_containers: Session,
) -> None:
    assert database_session_with_containers.scalar(text("SELECT 1")) == 1
    assert "sqlalchemy" in database_app_with_containers.extensions
    assert "redis" not in database_app_with_containers.extensions
    assert not database_app_with_containers.blueprints
