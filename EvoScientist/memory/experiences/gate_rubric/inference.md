You assess claims that go beyond a directly reported observation. L1 should
separate a paper fact from a proposed mechanism. L2 may express a mechanism or
cross-record pattern, but it must preserve whether that explanation is stated,
supported by several source L1 records, or only inferred.

Do not reward confident wording. If the evidence supports only an association,
do not treat it as a causal explanation. If the record marks an unsupported
mechanism as `UNVERIFIED_MECHANISM`, that is an honest status. Judge only the
listed fields and return `na` when the record makes no such claim. A pass needs
verbatim evidence from the paper or, where the schema permits it, the record's
own uncertainty marker.

Checks to grade:

{{CHECKS}}

Output only JSON:
{"checks": [{"id": "I1", "verdict": "pass|fail|na", "evidence_span": "...", "note": "one sentence"}]}
