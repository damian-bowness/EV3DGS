/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
					const float *orig_points,
					const glm::vec3 *scales,
					const glm::vec4 *rotations,
					const float *opacities,
					const float *shs,
					bool *clamped,
					const float *cov3D_precomp,
					const float *colors_precomp,
					const float *view2gaussian_precomp,
					const float *viewmatrix,
					const float *projmatrix,
					const glm::vec3 *cam_pos,
					const int W, int H,
					const float focal_x, float focal_y,
					const float tan_fovx, float tan_fovy,
					const float kernel_size,
					int *radii,
					float2 *points_xy_image,
					float *depths,
					float *cov3Ds,
					float *view2gaussians,
					float *quadricCoeffs,
					float *colors,
					float4 *conic_opacity,
					const dim3 grid,
					uint32_t *tiles_touched,
					bool prefiltered);

	// Main rasterization method.

	void computeRatio(
		int blocks, int threads,
		float *dCount,
		const float *dReject,
		int P);

	void gradient(
		const dim3 grid, dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		int W, int H,
		float xg_thresh,
		float focal_x, float focal_y,
		const float *features,
		const float *view2gaussian,
		const float *quadricCoeffs,
		const glm::vec4 *rotations,
		const float3 *means3D,
		const float3 *scales,
		const float4 *conic_opacity,
		float *usable_g, float *rejectCount);

	void rasterizeForward(
		const dim3 block,
		const dim3 tile_bounds,
		const dim3 img_size,
		const int32_t *__restrict__ gaussian_ids_sorted,
		const int2 *__restrict__ tile_bins,
		const float2 *__restrict__ xys,
		const float3 *__restrict__ conics,
		const float3 *__restrict__ colors,
		const float *__restrict__ opacities,
		float *__restrict__ final_Ts,
		int *__restrict__ final_index,
		float3 *__restrict__ out_img,
		const float3 &__restrict__ background);

	void renderRasterization(
		const float *usable_g,
		const bool two_pass, float cc_thresh,
		const dim3 grid, dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		int W, int H,
		const float2 *points_xy_image,
		const float *features,
		const float4 *conic_opacity,
		float *final_T,
		uint32_t *n_contrib,
		const float *bg_color,
		float *out_color,
		float *depths);

	void renderRayMarch(
		const float *usable_g,
		const dim3 grid, dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		int W, int H,
		const bool two_pass, float cc_thresh,
		float focal_x, float focal_y,
		const float *features,
		const float *quadricCoeffs,
		const glm::vec4 *rotations,
		const float3 *means3D,
		const float3 *scales,
		const float4 *conic_opacity,
		const float *bg_color,
		float *out_color);

	// Perform initial steps for each Point prior to integration.
	void preprocess_points(int PN, int D, int M,
						   const float *points3D,
						   const float *viewmatrix,
						   const float *projmatrix,
						   const glm::vec3 *cam_pos,
						   const int W, int H,
						   const float focal_x, float focal_y,
						   const float tan_fovx, float tan_fovy,
						   float2 *points2D,
						   float *depths,
						   const dim3 grid,
						   uint32_t *tiles_touched,
						   bool prefiltered);
}

#endif