# Testcontainers Integration Tests

Run a focused local test directly with pytest:

```bash
uv run --project api --dev pytest \
  api/tests/test_containers_integration_tests/path/to/test_file.py
```

Testcontainers tests reject xdist because every worker would create a separate container stack. Local Make targets also run without xdist. CI runs this suite once with automatic service selection, separately from the older Compose-backed integration tests.

## Container Selection

`--tc-services=auto` is the default. It always starts Postgres and adds other services from:

- `requires_redis`, `requires_sandbox`, and `requires_plugin_daemon` markers.
- Direct imports of the shared Redis client in selected test files.
- The code-executor test path, which implies Sandbox.

Plugin Daemon implies Redis. Services that are not selected receive inert local endpoints so application initialization remains deterministic without starting their containers.

Use `--tc-services=all` for compatibility runs or a comma-separated list such as `--tc-services=postgres,redis` for diagnosis. Mark the narrowest test that actually exercises an external service:

```python
@pytest.mark.requires_redis
def test_cache_invalidation(...):
    ...
```

## Database Isolation

The default remains post-test `TRUNCATE ... CASCADE`. Use it for paths that create transactions from `db.engine`, pass an engine into services, or otherwise escape the shared session scope.

Tests whose complete request path can share one connection should request `transactional_db_session`. It:

- Opens one outer transaction.
- Rebinds `db.session` and `core.db.session_factory` to that connection.
- Makes application commits and compatible `sessionmaker(db.engine)` work happen below a guard savepoint.
- Adds `no_container_truncate` automatically.
- Rolls back the outer transaction after the test.
- Fails if application code ends the outer transaction.

Use `database_state` for post-request reads because Flask request teardown removes the scoped session and detaches ORM objects:

```python
def test_create(
    authenticated_console_client: AuthenticatedConsoleClient,
    database_state: DatabaseState,
) -> None:
    tenant_id = authenticated_console_client.tenant.id

    with database_state.expect_count_change(MyModel, MyModel.tenant_id == tenant_id, before=0, after=1):
        response = authenticated_console_client.client.post(...)
        assert response.status_code == 201

    persisted = database_state.one(MyModel, MyModel.id == response.json["id"])
    assert persisted.tenant_id == tenant_id
```

Do not apply `no_container_truncate` alone unless another fixture fully owns cleanup. Prefer `transactional_db_session` so the isolation contract is executable.

## Console Fixtures

Console controller tests can use:

- `authenticated_console_client`: a real persisted account, tenant, owner membership, setup marker, JWT, and CSRF headers.
- `console_account_factory`: additional persisted account/tenant pairs with a selectable tenant role.
- `transactional_db_session`: setup and assertion session for savepoint-compatible tests.
- `database_state`: fresh post-request count, list, and single-row assertions.

Capture IDs and other scalar values before an HTTP request when retaining a setup ORM object. Request teardown deliberately detaches those objects.

## Migration Check

A controller test is ready for transaction isolation when every database path uses `db.session`, the injected `Session`, or `core.db.session_factory`. Keep truncate isolation and record a blocker when the path uses patterns such as:

- `with db.engine.begin()`
- `Session(db.engine)` followed by rollback-sensitive work
- An engine-typed service such as `FileService(db.engine)`
- A separately configured engine or session factory

The transaction fixture is a migration tool, not a way to conceal those ownership boundaries.
