"""Build a forward-dated held-out split and write it to JSONL."""

import latticebench as lb

records = lb.make_split(
    master_seed=20261231,
    n=4,
    m=4,
    count=100,
    created="2026-12-31",
    split="held-out-2026Q4",
)
written = lb.write_jsonl("data/held-out-2026Q4.jsonl", records)
print(f"wrote {written} puzzles")

manifest = lb.build_manifest(
    20261231,
    n=4,
    m=4,
    count=100,
    created="2026-12-31",
    split="held-out-2026Q4",
    generator_version=lb.__version__,
)
print("manifest fingerprint:", manifest.hash_of_hashes)
