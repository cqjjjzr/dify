from __future__ import annotations

from collections.abc import Generator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, cast

from flask import Flask
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session, sessionmaker

import core.db.session_factory as session_factory_module
from extensions.ext_database import db

_ModelT = TypeVar("_ModelT")


@contextmanager
def bind_test_transaction(app: Flask) -> Generator[Session, None, None]:
    with app.app_context():
        original_engine = db.engines[None]
        connection = original_engine.connect()
        outer_transaction = connection.begin()
        guard_savepoint = connection.begin_nested()
        engine_registry = cast(MutableMapping[str | None, object], db.engines)
        original_scoped_options = db.session.session_factory.kw.copy()
        original_session_maker = session_factory_module._session_maker

        db.session.remove()
        engine_registry[None] = connection
        db.session.session_factory.configure(join_transaction_mode="create_savepoint")
        session_factory_module._session_maker = sessionmaker(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

        try:
            yield db.session()
        finally:
            db.session.remove()
            session_factory_module._session_maker = original_session_maker
            db.session.session_factory.kw.clear()
            db.session.session_factory.kw.update(original_scoped_options)
            engine_registry[None] = original_engine

            transaction_was_active = outer_transaction.is_active and guard_savepoint.is_active
            if transaction_was_active:
                outer_transaction.rollback()
            connection.close()

            if not transaction_was_active:
                raise RuntimeError(
                    "The test's outer transaction was ended by application code. "
                    "Keep this test on truncate isolation until its session path is migrated."
                )


@dataclass(frozen=True)
class DatabaseState:
    session: Session

    def count(self, model: type[_ModelT], *criteria: ColumnElement[bool]) -> int:
        statement = select(func.count()).select_from(model)
        if criteria:
            statement = statement.where(*criteria)
        return self.session.scalar(statement) or 0

    def all(self, model: type[_ModelT], *criteria: ColumnElement[bool]) -> Sequence[_ModelT]:
        self.session.expire_all()
        statement = select(model)
        if criteria:
            statement = statement.where(*criteria)
        return self.session.scalars(statement).all()

    def one(self, model: type[_ModelT], *criteria: ColumnElement[bool]) -> _ModelT:
        self.session.expire_all()
        statement = select(model).where(*criteria)
        return self.session.scalars(statement).one()

    @contextmanager
    def expect_count_change(
        self,
        model: type[_ModelT],
        *criteria: ColumnElement[bool],
        before: int,
        after: int,
    ) -> Generator[None, None, None]:
        assert self.count(model, *criteria) == before
        yield
        self.session.expire_all()
        assert self.count(model, *criteria) == after
