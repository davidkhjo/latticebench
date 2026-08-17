# Contamination-free protocol and threat model

This document specifies how LatticeBench prevents benchmark contamination and,
just as important, states plainly what it does and does not guarantee. The
protocol is the reason a LatticeBench number can be trusted even after a benchmark
has been public for a year. It only works if every claim below holds, so we keep
the honest limits in the same document as the mechanism.

## The core idea

We do not distribute solved puzzles. We distribute a *manifest* from which anyone
can regenerate the exact instances, and we regenerate them again server-side when
a result is submitted. A leaked instance is worthless because the grader does not
trust submitted instances; it trusts the manifest and re-derives everything.

## The manifest

A manifest is a small, human-readable record with three load-bearing fields.

1. **Master seed.** A single integer that seeds the generator's RNG. Combined
   with a batch index, it determines every instance in the batch.

2. **Generator version.** The released version of the generation code. The mapping
   from seed to puzzle is only stable within a version, because a change to clue
   enumeration or minimization order changes the output. Pinning the version makes
   the mapping reproducible.

3. **Git commit.** The exact commit hash of the generator at release time. The
   version string is the human label; the commit is the cryptographic anchor that
   removes any ambiguity about which code produced the batch.

A manifest also records the batch configuration (grid sizes, clue budget, how
many instances) and the creation date discussed below. It contains no puzzles and
no solutions.

## Bit-for-bit regeneration

Given a manifest, the generator produces the identical batch on any machine. The
generation pipeline is a deterministic function of the seed and the code: all
randomness flows through one seeded RNG, serialization is canonical, and no step
depends on wall-clock time, hardware, or thread scheduling. We verify this in CI
by regenerating a reference batch and comparing hashes across platforms. If two
regenerations of the same manifest ever disagree, that is a release-blocking bug.

## Per-instance hashing

Every instance carries a hash of its canonical serialization, and the hash chains
back to the manifest: instance `i` is defined as the output of the generator on
`(master_seed, i, generator_version)`, and its recorded hash must match what that
call produces. This has two consequences. A submitter cannot swap in an easier
instance, because the swapped instance will not hash to the value the manifest
implies. And a submitter cannot hand-edit a hard instance into a solvable one,
because the edit changes the hash. Grading recomputes the instances from the
manifest and ignores whatever instances the submission claims to have used.

## Forward-dated held-out splits

Genuine contamination protection has to survive the fact that today's manifest
could be memorized by tomorrow's model. We handle this with creation dates. Each
instance and each manifest carry the date they were minted. A held-out split is a
manifest published on a known date; because instances are only ever produced by
running the pinned generator on the seed, a model whose training cutoff precedes
that date cannot have seen them. To evaluate such a model fairly, use a manifest
minted after its cutoff. The date is not a promise about the future; it is a
checkable fact about when the seed was published, and the leaderboard records it
next to every result so readers can judge the gap between a model's cutoff and the
manifest's date.

## Threat model

We enumerate the ways a result could be contaminated or gamed and how the
protocol answers each.

**Leaked instances.** Someone publishes solved puzzles from a batch. *Answered:*
grading re-derives instances from the manifest and re-grades server-side, so
memorizing the public batch only helps on that batch, and a fresh manifest voids
the advantage.

**Instance substitution.** A submitter runs on easier puzzles than the manifest
specifies. *Answered:* per-instance hashing chains each instance to
`(seed, index, version)`; a substituted instance fails the hash check.

**Answer injection.** A submitter edits a puzzle so its model looks correct.
*Answered:* the edit changes the hash and the re-derivation ignores the submitted
puzzle text entirely.

**Retroactive training.** A model is trained on a batch after it was published,
then reported as if solved cold. *Answered:* only partially, and honestly. The
creation date documents when the seed went public; it cannot prove a submitter
did not train on it afterward. Forward-dated manifests published after a model's
frozen cutoff are the real defense, and the leaderboard shows the dates so the
reader can apply skepticism where the gap is small or negative.

## Residual risk: genre familiarity is not instance contamination

The protocol guarantees that a specific instance was not in a model's training
data when the manifest postdates the cutoff. It does not, and cannot, guarantee
that the model has never seen a logic-grid puzzle. Models are broadly familiar
with the zebra-puzzle genre, its clue phrasings, and its solving strategies, and
that familiarity legitimately raises scores. We treat this as a feature of the
task rather than a leak, and we separate it from instance contamination in two
ways. First, we vary surface phrasing across a batch so that memorized clue
templates do not transfer. Second, we quantify the residual by comparing accuracy
on a fresh forward-dated manifest against accuracy on an older public one of the
same configuration: if genre familiarity were the whole story, the two would
match, and any gap upper-bounds instance-level contamination. We report that gap
rather than claiming it is zero.
