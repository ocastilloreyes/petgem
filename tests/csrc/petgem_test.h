/*
 * Filename: petgem_test.h
 * Author: PETGEM test suite
 * Date: 2026-07-03
 *
 * Description:
 * Minimal check/report macros shared by the C test harnesses (levels 1-3).
 * A harness runs every invariant, prints "<name>: N checks, F failures", and
 * returns non-zero when any check failed so pytest can assert on the exit code.
 * This is TEST INFRASTRUCTURE only; it links against the unchanged production
 * sources (fe_nedelec.c / fe_nodal.c / fem.c) to verify the real implementation.
 */
#ifndef PETGEM_TEST_H
#define PETGEM_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int pt_checks   = 0;
static int pt_failures = 0;

/* Boolean invariant with a printf-style message emitted on failure. */
#define PT_CHECK(cond, ...)                                                    \
  do {                                                                         \
    pt_checks++;                                                               \
    if (!(cond)) {                                                             \
      pt_failures++;                                                           \
      fprintf(stderr, "  FAIL [%s:%d] ", __FILE__, __LINE__);                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

/* |a - b| <= tol */
#define PT_CLOSE(a, b, tol, ...) PT_CHECK(fabs((double)(a) - (double)(b)) <= (double)(tol), __VA_ARGS__)

static int pt_report(const char *name)
{
  fprintf(stdout, "%s: %d checks, %d failures\n", name, pt_checks, pt_failures);
  fflush(stdout);
  return pt_failures ? 1 : 0;
}

#endif /* PETGEM_TEST_H */
