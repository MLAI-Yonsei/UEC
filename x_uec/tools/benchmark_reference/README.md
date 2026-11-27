These are the fixed IDs, exact text hashes, protocols, and encoding policies for
the exploratory English six-task pilot, not a full MTEB benchmark. No raw text or
embeddings are stored here. `manifest.json` seals the two runtime reference files.

`benchmark_inputs.py public-sources --output DATA --cache-dir HF_HUB` downloads
revision-pinned public sources, reconstructs original row order and text, and
verifies every row hash plus qrels, classification labels, and STS pair identity.
Classification reconstruction requires `datasets==2.21.0`; the pinned dataset
scripts are loaded with explicit trust, without an interactive prompt.

`benchmark_inputs.py public --output BENCHMARKS --cache-dir HF_HUB --device cuda:0`
also generates native means from the pinned model weights with each task's
original query/corpus/classification/STS prompt and sequence-length policy.
It checks every FP64 reconstructed mean against the actual encoder center and
seals independent row IDs and runtime provenance. No Gaussian is fitted and no
variance or previous alignment map is read. An interrupted role resumes from
completed role seals; an interrupted role is recomputed.

The public encoding uses the current final-path runtime. It is not asserted to
be bit-identical to the original native caches: those Qwen arrays used
flash_attention_2 with FP16 output normalization, whereas the default current
runtime uses sdpa and the reconstructed FP64 final path. Those differences and
the effective role prompts are recorded in each task's provenance.

`legacy-import` is an optional, explicit importer for already certified caches;
it verifies the original file hashes, encoder policy, model-specific row IDs,
exact text hashes, cache keys, and array contracts before exporting float64
row-unit arrays. It never imports private experiment packages.

Text overlap grouping uses NFC plus CRLF/CR-to-LF only. The historical
`canonical_hash` fields collapsed whitespace; they are preserved as provenance,
while `canonical_text_sha256` implements the current alignment exclusion rule.

`validate_benchmark_suite(BENCHMARKS)` is the canonical scoring boundary. A seal
alone is insufficient: it also requires the exact six published tasks, row IDs
and text hashes, source/probe protocols, native model roles, and policies. Legacy
imports require the certified cache-index hash and all six original source
manifest hashes. Public generation binds request mode, packaged reference,
encoding implementation, frozen producers, attention/device/dtype, library
versions, physical batch, actual model policies, and independent IDs at role,
task, and suite publication; incompatible resumptions fail before scoring.
