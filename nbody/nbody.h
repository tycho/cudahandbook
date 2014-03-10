/*
 *
 * nbody.h
 *
 * Header file to declare globals in nbody.cu
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __CUDAHANDBOOK_NBODY_H__
#define __CUDAHANDBOOK_NBODY_H__

#include "nbody_CPU_SIMD.h"

extern float *g_hostAOS_PosMass;
extern float *g_hostAOS_VelInvMass;
extern float *g_hostAOS_Force;

extern float *g_hostSOA_Pos[3];
extern float *g_hostSOA_Force[3];
extern float *g_hostSOA_Mass;
extern float *g_hostSOA_InvMass;

// maximum number of GPUs supported by single-threaded multi-GPU
const int g_maxGPUs = 32;

extern int g_numCPUCores;
extern int g_numGPUs;

extern float ComputeGravitation_GPU_Shared( float *force, float const * const posMass, float softeningSquared, size_t N );
extern float ComputeGravitation_multiGPU  ( float *force, float const * const posMass, float softeningSquared, size_t N );

#endif

/* vim: set ts=4 sts=4 sw=4 et: */
