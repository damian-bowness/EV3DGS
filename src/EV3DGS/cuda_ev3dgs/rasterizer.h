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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static void view2gaussian(
			const int num_points,
			const float *means3D,
			const float *rotations,
			const float *scales,
			const float *viewmatrix,
			float *view2gaussianMat,
			float *quadricCoeffs);

		static void rasterizeForward(
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

		static void forward(
			std::function<char *(size_t)> geometryBuffer,
			std::function<char *(size_t)> binningBuffer,
			std::function<char *(size_t)> imageBuffer,
			const int P, int D, int M,
			const float *background,
			const int width, int height,
			const bool rastFlag, bool two_pass,
			const float xg_thresh, float cc_thresh,
			const float *means3D,
			const float *shs,
			const float *colors_precomp,
			const float *opacities,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const int *gaussian_index,
			const float *cov3D_precomp,
			const float *view2gaussian_precomp,
			const float *viewmatrix,
			const float *projmatrix,
			const float *cam_pos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const float *subpixel_offset,
			const bool prefiltered,
			float *out_color,
			int *radii = nullptr,
			bool debug = false);
	};
}

#endif