# AutoSOTA repository integration

**Archived: November 7, 2026.** Source package:
`AutoSOTA_GRPO_Research_Package_20261107.zip`.

The English report, source overlay, selected runtime records, reproduction guide,
result tables, 100-run ledger, licenses, and PNG figure are preserved. The duplicate
PDF figure is omitted and the README's asset description is adjusted accordingly.
All Python and shell implementations, overlay hashes, and selected settings are
unchanged.

PDA-Iso is an accuracy–length trade-off: compared with matched DrGRPO, it uses
56.91% fewer mean output tokens but answers nine fewer of 1,024 questions correctly.
This is not an accuracy improvement over DrGRPO or an equivalent reduction in
training time. The main confirmation uses one seed and common Hint/SFT conditions;
the report preserves those qualifications and the unsuccessful candidates.

The integration verifies result arithmetic and overlay hashes and runs the eight
NumPy-only adaptive-Hint unit tests. PyTorch-dependent suites, upstream framework
installation, distributed training, and model evaluation were not run on this
machine. The source overlay still requires its documented pinned upstream checkout.

See the common [source manifest](../source_manifest.json) and
[integration validation](../integration_validation.json) for source identities and
checks performed here.
