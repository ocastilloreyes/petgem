/*
 * Filename: receivers.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-09-05
 *
 * Description:
 * This file contains a collection of definitions for
 * receivers functions that are used throughout the PETGEM
 * project.
 *
 * Usage:
 * Include this file in your source code to utilize the
 * common functions. For example: #include "receivers.h"
 *
 */

#ifndef RECEIVER_H
#define RECEIVER_H

PetscErrorCode setupReceivers(petgemSource* source, int mode);

#endif