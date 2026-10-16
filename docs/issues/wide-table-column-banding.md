# Wide-table column values

Status: user-approved repair applied; both new table-format regressions and all 54 chunking tests pass in the full-suite run.

Tables with more than 25 columns are split into column groups, each retaining column zero as its key. Previously, row values retained their original global indexes while the renderers looked up local indexes. Values in later groups were silently omitted in both row-record and pipe formats.

The repair renumbers each selected row's columns locally before rendering. Column names and values now align in every group. A 30-column regression checks that every value survives, each group repeats its key and source block identifiers are preserved. Both formats are tested.

The complete 54-case chunking suite passes against the proposed patch in a temporary checkout. The user explicitly approved applying this correction as an exception to the implementation plan's algorithm-preservation constraint. Production coverage measurement remains unchanged in scope.
