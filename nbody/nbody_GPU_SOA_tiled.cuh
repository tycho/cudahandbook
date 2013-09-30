/*
 *
 * nbody_GPU_AOS_tiled.cuh
 *
 * CUDA implementation of the O(N^2) N-body calculation.
 * Tiled to take advantage of the symmetry of gravitational
 * forces: Fij=-Fji
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

template<size_t nTile>
__device__ void
DoDiagonalTile_GPU_SOA(
    float *forceX, float *forceY, float *forceZ,
    float const * const posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    int laneid = threadIdx.x&0x1f;
    size_t i = iTile*nTile+laneid;
    float acc[3] = {0, 0, 0};
    float myX = posMass[i*4+0];
    float myY = posMass[i*4+1];
    float myZ = posMass[i*4+2];

    for ( size_t _j = 0; _j < nTile; _j++ ) {
        size_t j = jTile*nTile+_j;

        float fx, fy, fz;
        float4 body = ((float4 *) posMass)[j];

        bodyBodyInteraction<float>(
            &fx, &fy, &fz,
            myX, myY, myZ,
            body.x, body.y, body.z, body.w,
            softeningSquared );
        acc[0] += fx;
        acc[1] += fy;
        acc[2] += fz;
    }

    atomicAdd( &forceX[i], acc[0] );
    atomicAdd( &forceY[i], acc[1] );
    atomicAdd( &forceZ[i], acc[2] );
}

template<size_t nTile>
__device__ void
DoNondiagonalTile_GPU_SOA(
    float *forceX, float *forceY, float *forceZ,
    float const * const posMass,
    float softeningSquared,
    size_t iTile, size_t jTile
)
{
    int laneid = threadIdx.x&0x1f;
    size_t i = iTile*nTile+laneid;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float4 myPosMass = ((float4 *) posMass)[i];
    float myX = myPosMass.x;
    float myY = myPosMass.y;
    float myZ = myPosMass.z;

    float4 shufSrcPosMass = ((float4 *) posMass)[jTile*nTile+laneid];

//#pragma unroll
    for ( size_t _j = 0; _j < nTile; _j++ ) {
        const size_t j = jTile*nTile+_j;

        float fx, fy, fz;
        float4 bodyPosMass;

        bodyPosMass.x = __shfl( shufSrcPosMass.x, _j );
        bodyPosMass.y = __shfl( shufSrcPosMass.y, _j );
        bodyPosMass.z = __shfl( shufSrcPosMass.z, _j );
        bodyPosMass.w = __shfl( shufSrcPosMass.w, _j );

        bodyBodyInteraction<float>(
            &fx, &fy, &fz,
            myX, myY, myZ,
            bodyPosMass.x, bodyPosMass.y, bodyPosMass.z, bodyPosMass.w,
            softeningSquared );

        ax += fx;
        ay += fy;
        az += fz;

        fx = warpReduce( -fx );
        fy = warpReduce( -fy );
        fz = warpReduce( -fz );

        if ( laneid == 0 ) {
            atomicAdd( &forceX[j], fx );
            atomicAdd( &forceY[j], fy );
            atomicAdd( &forceZ[j], fz );
        }
    }

    atomicAdd( &forceX[i], ax );
    atomicAdd( &forceY[i], ay );
    atomicAdd( &forceZ[i], az );

}

template<size_t nTile>
__global__ void
ComputeNBodyGravitation_GPU_SOA_tiled(
    float *forceX, float *forceY, float *forceZ,
    float const * const posMass,
    size_t N,
    float softeningSquared )
{
    int warpsPerBlock = nTile/32;
    const int warpid = threadIdx.x >> 5;

    int iTileCoarse = blockIdx.x;
    int iTile = iTileCoarse*warpsPerBlock+warpid;
    int jTile = blockIdx.y;

    if ( iTile == jTile ) {
        DoDiagonalTile_GPU_SOA<32>( forceX, forceY, forceZ, posMass, softeningSquared, iTile, jTile );
    }
    else if ( jTile < iTile ) {
        DoNondiagonalTile_GPU_SOA<32>( forceX, forceY, forceZ, posMass, softeningSquared, iTile, jTile );
    }
}

template<size_t nTile>
cudaError_t
ComputeGravitation_GPU_SOA_tiled(
    float *forces[3],
    float const * const posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    dim3 blocks( N/nTile, N/32, 1 );

    CUDART_CHECK( cudaMemset( forces[0], 0, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( forces[1], 0, N*sizeof(float) ) );
    CUDART_CHECK( cudaMemset( forces[2], 0, N*sizeof(float) ) );
    ComputeNBodyGravitation_GPU_SOA_tiled<nTile><<<blocks,nTile>>>( forces[0], forces[1], forces[2], posMass, N, softeningSquared );
    CUDART_CHECK( cudaDeviceSynchronize() );
Error:
    return status;
}

__global__ void
AOStoSOA_GPU_3( float *outX, float *outY, float *outZ, const float *in, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        float tmp[3] = { in[i*3+0], in[i*3+1], in[i*3+2] };
        outX[i] = tmp[0];
        outY[i] = tmp[1];
        outZ[i] = tmp[2];
    }
}

__global__ void
SOAtoAOS_GPU_3( float *out, const float *inX, const float *inY, const float *inZ, size_t N )
{
    for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
                 i < N;
                 i += blockDim.x*gridDim.x ) {
        out[3*i+0] = inX[i];
        out[3*i+1] = inY[i];
        out[3*i+2] = inZ[i];
    }
}

float
ComputeGravitation_GPU_SOA_tiled(
    float *force,
    float const * const posMass,
    float softeningSquared,
    size_t N
)
{
    cudaError_t status;
    cudaEvent_t evStart = 0, evStop = 0;
    float ms = 0.0;

float *forces[3] = {0};
CUDART_CHECK( cudaMalloc( &forces[0], N*sizeof(float) ) );
CUDART_CHECK( cudaMalloc( &forces[1], N*sizeof(float) ) );
CUDART_CHECK( cudaMalloc( &forces[2], N*sizeof(float) ) );

    CUDART_CHECK( cudaEventCreate( &evStart ) );
    CUDART_CHECK( cudaEventCreate( &evStop ) );

AOStoSOA_GPU_3<<<300,256>>>( forces[0], forces[1], forces[2], force, N );

    CUDART_CHECK( cudaEventRecord( evStart, NULL ) );
    CUDART_CHECK( ComputeGravitation_GPU_SOA_tiled<128>(
        forces,
        posMass,
        softeningSquared,
        N ) );
    CUDART_CHECK( cudaEventRecord( evStop, NULL ) );

    CUDART_CHECK( cudaDeviceSynchronize() );
SOAtoAOS_GPU_3<<<300,256>>>( force, forces[0], forces[1], forces[2], N );


    CUDART_CHECK( cudaDeviceSynchronize() );
    CUDART_CHECK( cudaEventElapsedTime( &ms, evStart, evStop ) );
Error:
    CUDART_CHECK( cudaEventDestroy( evStop ) );
    CUDART_CHECK( cudaEventDestroy( evStart ) );
    return ms;
}

/* vim: set ts=4 sts=4 sw=4 et: */
