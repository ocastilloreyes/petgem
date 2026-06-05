#!/usr/bin/env bash
#
# check_doc_drift.sh - guard against documentation / inline-comment drift in
# the PETGEM C sources. Two independent, fast, PETSc-free checks:
#
#   1. Removed-symbol guard. Fails if a retired CODE token reappears. These
#      tokens are never legitimate (current baseline is zero), so any hit is
#      real drift:
#        - MatFilter()            G_BDDC uses MAT_IGNORE_ZERO_ENTRIES instead
#        - <imParams>->nord       must be ->fm.nord (nord lives in embedded fm)
#        - <params>->bundleFile   removed struct field (use ->fm.inputFile)
#        - /inv_sources           removed HDF5 group (unified into /sources)
#        - -fm_grad_check         removed diagnostic option
#      Regexes match the CODE form only, so legitimate prose - e.g. the
#      "no MatFilter pass is required" note in assembly.c or the historical
#      "canonical gradient" note in solver.c - does NOT trip the guard.
#
#   2. Doxygen @param consistency. If doxygen is installed, parses the
#      sources and fails on "is not found in the argument list" warnings,
#      i.e. a docstring @param that no longer matches its function signature
#      (the inversion.h:198 / createInvKSP class of drift). Pre-existing
#      "undocumented" warnings are intentionally ignored so the gate stays
#      focused on drift, not coverage.
#
# Prose drift (a comment that mis-describes behaviour without using a retired
# token, e.g. "five cell-centered fields" when only one is written) is NOT
# machine-detectable here and still relies on code review.
#
# Usage:   bash scripts/auto_doc/check_doc_drift.sh
# Exit:    0 = clean, 1 = drift detected.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

fail=0

# ---------------------------------------------------------------------------
# Check 1 - removed-symbol guard
# ---------------------------------------------------------------------------
# Entries are "<extended-regex>|<human description>". The regexes contain no
# literal '|', so the field split below is unambiguous.
GUARDS=(
  'MatFilter\(|MatFilter() call - G_BDDC relies on MAT_IGNORE_ZERO_ENTRIES'
  '[iI]m?_?[Pp]arams->nord\b|imParams->nord - use ->fm.nord (nord is in the embedded fmParams)'
  '[Pp]arams->bundleFile\b|removed struct field bundleFile - use ->fm.inputFile'
  '/inv_sources\b|removed HDF5 group /inv_sources - unified into /sources'
  '_fm_grad_check\b|removed diagnostic option -fm_grad_check'
)

echo "== check_doc_drift: removed-symbol guard =="
for g in "${GUARDS[@]}"; do
  regex="${g%%|*}"
  desc="${g##*|}"
  hits="$(grep -rInE "$regex" src/ include/ 2>/dev/null || true)"
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
  # Layer output-suppressing overrides on top of the repo Doxyfile and feed
  # the result to doxygen via stdin ('-'). We only want the warning stream.
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
