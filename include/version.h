/*
 * Filename: version.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * PETGEM version macros (VERSION_MAJOR / VERSION_MINOR / VERSION_PATCH).
 */

#ifndef VERSION_H
#define VERSION_H

/**
 * @brief PETGEM semantic version components.
 *
 * Reported by each kernel's `--version` handler and stamped into the
 * provenance attributes of HDF5 output files.
 */
#define VERSION_MAJOR 2
#define VERSION_MINOR 0
#define VERSION_PATCH 0

#endif
