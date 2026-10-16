# Search filters can override tenant and job scope

**Status:** User-authorized repair applied; the full 448-test suite passes.

**Detected during:** User-requested whole-project test expansion.

**Affected interface:** `POST /search`, `src/longparser/server/app.py`.

The endpoint validates the requested job against the authenticated tenant, but merges caller-supplied filters after the trusted `tenant_id` and `job_id`. A caller can therefore replace those two vector-search constraints. The HTTP regression proves that the Chroma SDK receives the attacker-supplied tenant/job values. This establishes an isolation defect in query construction; it does not claim a live-service data breach.

## Evidence

`test_search_filters_cannot_override_authenticated_tenant_or_job` sends a legitimate authenticated job with `filters={"tenant_id": "another-tenant", "job_id": "another-job"}`. The unmodified application sends those two values to vector search, and the expected-isolation assertion fails. Ordinary search and custom non-scope filters pass.

## Applied minimal repair

```python
filters = {
    **body.filters,
    "tenant_id": tenant_id,
    "job_id": body.job_id,
}
```

Trusted scope wins; other custom filters remain supported. No route, schema, provider or vector-store interface changes. The exact diff is prepared at `/tmp/longparser-search-scope-proposed.patch`; the same regression passes in `/tmp/longparser-search-scope-review` with that patch only.

The earlier implementation plan §12.3 protects server “Routes, RBAC/tenancy/CORS behavior.” The new user request authorizes broad testing, so the production tenancy repair is presented as a concrete scope exception. The failing assertion stays active; no skip or expected-failure marker hides it.

## Final disposition

The user explicitly requested resolving these errors, authorizing the plan exception. The production filter merge now gives authenticated tenant/job scope precedence. The HTTP regression and the complete default suite pass; whole-production coverage is 95.97%. No live-service breach is claimed.
