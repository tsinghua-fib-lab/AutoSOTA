# AutoSOTA repository integration

**Archived: November 7, 2026.** Source package:
`AutoSOTA_Skill_Evolve_case.zip`.

The implementation, final skills, MIT license, reported stage comparison, and
saved evaluation records are preserved. All Python files and final skill texts
are unchanged. A study README and an evidence-coverage note in `final_report.md`
clarify the scope of the handoff.

Three Chinese comments in `best_code/configs/algo02/searchqa.yaml` are translated
into English without changing configuration values. In five evidence JSON files,
the Chinese component of historical machine paths is replaced with the English
alias `ai-cases`. Those paths remain provenance labels, not portable reproduction
paths; scores, identifiers, hashes, and metric fields are unchanged. Original file
hashes are recorded in the common source manifest.

Per-example records reproduce the final SearchQA, DocVQA, and ALFWorld endpoints;
their saved final skills match the recorded skill hashes. SpreadsheetBench's
reported endpoint has no accompanying final skill or per-example evaluation in
this package. Historical records refer to multiple code revisions, while
`best_code/` contains the latest supplied implementation.

Integration did not call a model provider or rerun the agent benchmarks. See the
common [source manifest](../source_manifest.json) and
[integration validation](../integration_validation.json) for file identities and
checks performed here.
