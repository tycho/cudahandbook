/*
 *
 * nbody_CPU_SOA_tiled.cpp
 *
 * Scalar CPU implementation of the O(N^2) N-body calculation.
 * Performs the computation in 128x128 tiles.
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

template<size_t nTile>
static void
DoDiagonalTile(
    float *force[3],
    float const * const pos[3],
    float const * const mass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    for ( size_t _i = 0; _i < nTile; _i++ )
    {
        const size_t i = iTile*nTile+_i;
        float acx, acy, acz;
        const float myX = pos[0][i];
        const float myY = pos[1][i];
        const float myZ = pos[2][i];

        acx = acy = acz = 0;

        #pragma simd vectorlengthfor(float) \
            reduction(+:acx) \
            reduction(+:acy) \
            reduction(+:acz)
        for ( size_t _j = 0; _j < nTile; _j++ ) {
            const size_t j = jTile*nTile+_j;

            float fx, fy, fz;
            const float bodyX = pos[0][j];
            const float bodyY = pos[1][j];
            const float bodyZ = pos[2][j];
            const float bodyMass = mass[j];

            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );

            acx += fx;
            acy += fy;
            acz += fz;
        }

        #pragma omp atomic update
        force[0][i] += acx;
        #pragma omp atomic update
        force[1][i] += acy;
        #pragma omp atomic update
        force[2][i] += acz;
    }
}

template<size_t nTile>
static void
DoNondiagonalTile(
    float *force[3],
    float const * const pos[3],
    float const * const mass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    float symmetricX[nTile];
    float symmetricY[nTile];
    float symmetricZ[nTile];

    memset( symmetricX, 0, sizeof(symmetricX) );
    memset( symmetricY, 0, sizeof(symmetricY) );
    memset( symmetricZ, 0, sizeof(symmetricZ) );

    for ( size_t _i = 0; _i < nTile; _i++ )
    {
        const size_t i = iTile*nTile+_i;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        const float myX = pos[0][i];
        const float myY = pos[1][i];
        const float myZ = pos[2][i];

        for ( size_t _j = 0; _j < nTile; _j++ ) {
            const size_t j = jTile*nTile+_j;

            float fx, fy, fz;
            const float bodyX = pos[0][j];
            const float bodyY = pos[1][j];
            const float bodyZ = pos[2][j];
            const float bodyMass = mass[j];

            bodyBodyInteraction<float>(
                &fx, &fy, &fz,
                myX, myY, myZ,
                bodyX, bodyY, bodyZ, bodyMass,
                softeningSquared );

            ax += fx;
            ay += fy;
            az += fz;

            symmetricX[_j] -= fx;
            symmetricY[_j] -= fy;
            symmetricZ[_j] -= fz;
        }

        #pragma omp atomic update
        force[0][i] += ax;
        #pragma omp atomic update
        force[1][i] += ay;
        #pragma omp atomic update
        force[2][i] += az;
    }

    for ( size_t _j = 0; _j < nTile; _j++ ) {
        const size_t j = jTile*nTile+_j;
        #pragma omp atomic update
        force[0][j] += symmetricX[_j];
        #pragma omp atomic update
        force[1][j] += symmetricY[_j];
        #pragma omp atomic update
        force[2][j] += symmetricZ[_j];
    }
}

template<size_t nTile>
static float
ComputeGravitation_SOA_tiled(
    float *force[3],
    float const * const pos[3],
    float const * const mass,
    float softeningSquared,
    size_t N
)
{
    memset( force[0], 0, N * sizeof(float) );
    memset( force[1], 0, N * sizeof(float) );
    memset( force[2], 0, N * sizeof(float) );
    chTimerTimestamp start, end;
    chTimerGetTime( &start );
    for ( size_t iTile = 0; iTile < N/nTile; iTile++ ) {
        #pragma omp parallel for
        for ( size_t jTile = 0; jTile < iTile; jTile++ ) {
            DoNondiagonalTile<nTile>(
                force,
                pos,
                mass,
                softeningSquared,
                iTile, jTile );
        }
    }
    #pragma omp parallel for
    for ( size_t iTile = 0; iTile < N/nTile; iTile++ ) {
        DoDiagonalTile<nTile>(
            force,
            pos,
            mass,
            softeningSquared,
            iTile, iTile );
    }
    chTimerGetTime( &end );
    return (float) chTimerElapsedTime( &start, &end ) * 1000.0f;
}

float
ComputeGravitation_SOA_tiled(
    float *force[3],
    float const * const pos[3],
    float const * const mass,
    float softeningSquared,
    size_t N
)
{
    return ComputeGravitation_SOA_tiled<128>(
        force,
        pos,
        mass,
        softeningSquared,
        N );
}

/* vim: set ts=4 sts=4 sw=4 et: */
