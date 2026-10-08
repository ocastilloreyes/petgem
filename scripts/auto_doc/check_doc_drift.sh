#!/usr/bin/env bash

# ============================================================================
# Documentation drift check
#
# This script checks that the documentation and comments in the PETGEM C
# sources still match the code. It runs two quick checks that do not need
# PETSc:
#
#   1. Retired names. It fails if a removed symbol, HDF5 group or option
#      comes back into src/ or include/ (the list is in GUARDS below). Only
#      the code form is matched, so prose that mentions an old name passes.
#
#   2. Doxygen @param names. When doxygen is installed, it fails if a
#      documented @param no longer matches the function signature. Warnings
#      about undocumented items are ignored on purpose.
#
# Comments that describe the behaviour wrongly without using a retired name
# cannot be detected here and still need code review.
#
# Usage:
#   bash scripts/auto_doc/check_doc_drift.sh
#
# Exit status:
#   0  no drift found
#   1  drift found
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

fail=0

# ---------------------------------------------------------------------------
# Check 1 - removed-symbol guard
# ---------------------------------------------------------------------------
# Each entry is "<extended regex>|<description>". The regexes contain no
# literal '|', so splitting on it is safe.
GUARDS=(
  '[iI]m?_?[Pp]arams->nord\b|imParams->nord - use ->fm.nord (nord is in the embedded fmParams)'
  '[Pp]arams->bundleFile\b|removed struct field bundleFile - use ->fm.inputFile'
  '/inv_sources\b|removed HDF5 group /inv_sources - unified into /sources'
  '_fm_grad_check\b|removed diagnostic option -fm_grad_check'
  '"-inv_[a-z_]*"|retired option namespace -inv_* - every inversion option is now -im_*'
  '/inv_meta\b|renamed HDF5 group /inv_meta - now /im_meta'
  '"Nord"|retired output attribute "Nord" - the unified provenance block writes "order"'
  '"Petgem_version"|retired output attribute "Petgem_version" - now lowercase "petgem_version"'
  'inv_model_iter|retired VTU snapshot prefix - snapshots derive their stem from -output_filename'
  '\bfmParams\b|renamed type fmParams - the shared base is petgemParams (fm.csem uses it directly, imParams embeds it as .common)'
  '\breadfmParams\b|renamed function readfmParams - now readPetgemParams (it parses the options common to BOTH kernels)'
  '\bINV_(MAX_FIXED_MATERIALS|MAX_FREQUENCIES|VTU_NUM_FIELDS)\b|retired INV_* constants - now IM_*'
)

echo "== check_doc_drift: removed-symbol guard =="
for g in "${GUARDS[@]}"; do
  regex="${g%%|*}"
  desc="${g##*|}"
  # Lines tagged DRIFT_GUARD_ALLOW are skipped: they belong to the code that
  # rejects the retired names (readimParams).
  hits="$(grep -rInE "$regex" src/ include/ 2>/dev/null \
            | grep -v 'DRIFT_GUARD_ALLOW' || true)"
  if [ -n "$hits" ]; then
    echo "  DRIFT - $desc"
    echo "$hits" | sed 's/^/      /'
    fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "  OK - no retired tokens found."

# ---------------------------------------------------------------------------
# Check 2 - Doxygen @param-vs-signature consistency (best effort)
# ---------------------------------------------------------------------------
echo "== check_doc_drift: Doxygen @param consistency =="
if command -v doxygen >/dev/null 2>&1; then
  warn_log="$(mktemp)"
  # Run doxygen on the repository Doxyfile with all output disabled; only
  # the warnings are needed.
  {
    cat Doxyfile
    printf '%s\n' \
      "GENERATE_HTML=NO" "GENERATE_LATEX=NO" "GENERATE_XML=NO" \
      "GENERATE_RTF=NO"  "GENERATE_MAN=NO"   "QUIET=YES" \
      "WARN_LOGFILE=$warn_log"
  } | doxygen - >/dev/null 2>&1 || true

  mismatches="$(grep -E "is not found in the argument list" "$warn_log" 2>/dev/null || true)"
  rm -f "$warn_log"
  if [ -n "$mismatches" ]; then
    echo "  DRIFT - @param names that no longer match a signature:"
    echo "$mismatches" | sed 's/^/      /'
    fail=1
  else
    echo "  OK - every documented @param matches its function signature."
  fi
else
  echo "  SKIP - doxygen not installed (install it to enable @param checks)."
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "check_doc_drift: PASS"
else
  echo "check_doc_drift: FAIL - documentation drift detected (see above)."
fi
exit "$fail"
