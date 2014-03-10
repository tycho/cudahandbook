/*
 *
 * nbody_CPU_SOA.h
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
 * This SOA (structure of arrays) formulation blazes the trail
 * for an SSE implementation.
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

#ifndef NO_CUDA
#define NO_CUDA
#endif
#include <chCUDA.h>
#include <chTimer.h>

#include "bodybodyInteraction.cuh"
#include "nbody_CPU_SOA.h"

float
ComputeGravitation_SOA(
    float *force[3],
    float const * const pos[3],
    float const * const mass,
    float softeningSquared,
    size_t N
)
{
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
#pragma omp parallel for
    for ( size_t i = 0; i < N; i++ )
    {
        float acx, acy, acz;
        const float myX = pos[0][i];
        const float myY = pos[1][i];
        const float myZ = pos[2][i];

        acx = acy = acz = 0;

        #pragma simd vectorlengthfor(float) \
            reduction(+:acx) \
            reduction(+:acy) \
            reduction(+:acz)
        for ( size_t j = 0; j < N; j++ ) {

            const float bodyX = pos[0][j];
            const float bodyY = pos[1][j];
            const float bodyZ = pos[2][j];
            const float bodyMass = mass[j];

            float fx, fy, fz;
            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );

            acx += fx;
            acy += fy;
            acz += fz;
        }

        force[0][i] = acx;
        force[1][i] = acy;
        force[2][i] = acz;
    }
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}

/* vim: set ts=4 sts=4 sw=4 et: */
