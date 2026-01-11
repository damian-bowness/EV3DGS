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
#include <iostream>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
					 SH_C2[0] * xy * sh[4] +
					 SH_C2[1] * yz * sh[5] +
					 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					 SH_C2[3] * xz * sh[7] +
					 SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
						 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						 SH_C3[1] * xy * z * sh[10] +
						 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						 SH_C3[5] * z * (xx - yy) * sh[14] +
						 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float4 computeCov2D(const float3 &mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, float kernel_size, const float *cov3D, const float *viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.

	// compute the coef of alpha based on the detemintant
	const float det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
	const float det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
	float coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);

	if (det_0 <= 1e-6 || det_1 <= 1e-6)
	{
		coef = 0.0f;
	}

	cov[0][0] += kernel_size;
	cov[1][1] += kernel_size;

	return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(coef)};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, const glm::vec4 rot, float *cov3D)
{
	// Create scaling matrix - can multiply with a modifier to tweak covariance matrix but I never do.
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Forward method for computing the inverse of the cov3D matrix
__device__ void computeCov3DInv(const float *cov3D, const float *viewmatrix, float *inv_cov3D)
{
	// inv cov before applying J
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 cov3D_view = glm::transpose(W) * glm::transpose(Vrk) * W;
	glm::mat3 inv = glm::inverse(cov3D_view);

	// inv_cov3D is in row-major order
	// since inv is symmetric, row-major order is the same as column-major order
	inv_cov3D[0] = inv[0][0];
	inv_cov3D[1] = inv[0][1];
	inv_cov3D[2] = inv[0][2];
	inv_cov3D[3] = inv[1][0];
	inv_cov3D[4] = inv[1][1];
	inv_cov3D[5] = inv[1][2];
	inv_cov3D[6] = inv[2][0];
	inv_cov3D[7] = inv[2][1];
	inv_cov3D[8] = inv[2][2];
}

__device__ glm::mat3 quat2rot(const glm::vec4 q)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	return R;
}

__device__ glm::mat4 computeGaussian2World(const float3 &mean, const glm::vec4 rot)
{

	// Compute rotation matrix from quaternion
	glm::mat3 R = quat2rot(rot);

	// transform 3D points in gaussian coordinate system to world coordinate system as follows
	// new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
	// so the rots is the gaussian to world transform

	// Gaussian to world transform - GLM is column major therefore use transpose to align with GLM conventions
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f);

	return G2W;
}

// TODO combined with computeCov3D to avoid redundant computation
// Forward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian(const glm::vec3 scale, const float3 &mean, const glm::vec4 rot, const float *viewmatrix, float *view2gaussian, float *quadricCoeffs)
{
	glm::mat4 G2W = computeGaussian2World(mean, rot);

	// could be simplied by using pointer
	// viewmatrix is the world to view transformation matrix
	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	// inverse of Gaussian to view transform
	// glm::mat4 V2G_inverse = glm::inverse(G2V);
	// R = G2V[:, :3, :3]
	// t = G2V[:, :3, 3]

	// t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
	// V2G = torch.zeros((N, 4, 4), device='cuda')
	// V2G[:, :3, :3] = R.transpose(1, 2)
	// V2G[:, :3, 3] = t2
	// V2G[:, 3, 3] = 1.0
	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;

	// write to view2gaussian
	view2gaussian[0] = R_transpose[0][0];
	view2gaussian[1] = R_transpose[0][1];
	view2gaussian[2] = R_transpose[0][2];
	view2gaussian[3] = 0.0f;
	view2gaussian[4] = R_transpose[1][0];
	view2gaussian[5] = R_transpose[1][1];
	view2gaussian[6] = R_transpose[1][2];
	view2gaussian[7] = 0.0f;
	view2gaussian[8] = R_transpose[2][0];
	view2gaussian[9] = R_transpose[2][1];
	view2gaussian[10] = R_transpose[2][2];
	view2gaussian[11] = 0.0f;
	view2gaussian[12] = t2.x;
	view2gaussian[13] = t2.y;
	view2gaussian[14] = t2.z;
	view2gaussian[15] = 1.0f;

	double3 S_inv_square = {1.0f / ((double)scale.x * scale.x + 1e-7), 1.0f / ((double)scale.y * scale.y + 1e-7), 1.0f / ((double)scale.z * scale.z + 1e-7)};
	double C = t2.x * t2.x * S_inv_square.x + t2.y * t2.y * S_inv_square.y + t2.z * t2.z * S_inv_square.z;
	glm::mat3 S_inv_square_R = glm::mat3(
		S_inv_square.x * R_transpose[0][0], S_inv_square.y * R_transpose[0][1], S_inv_square.z * R_transpose[0][2],
		S_inv_square.x * R_transpose[1][0], S_inv_square.y * R_transpose[1][1], S_inv_square.z * R_transpose[1][2],
		S_inv_square.x * R_transpose[2][0], S_inv_square.y * R_transpose[2][1], S_inv_square.z * R_transpose[2][2]);

	glm::vec3 B = t2 * S_inv_square_R;

	glm::mat3 Sigma = glm::transpose(R_transpose) * S_inv_square_R;

	// write to quadricCoeffs
	quadricCoeffs[0] = Sigma[0][0];
	quadricCoeffs[1] = Sigma[0][1];
	quadricCoeffs[2] = Sigma[0][2];
	quadricCoeffs[3] = Sigma[1][1];
	quadricCoeffs[4] = Sigma[1][2];
	quadricCoeffs[5] = Sigma[2][2];
	quadricCoeffs[6] = B.x;
	quadricCoeffs[7] = B.y;
	quadricCoeffs[8] = B.z;
	quadricCoeffs[9] = C;
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
							   const float tan_fovx, float tan_fovy,
							   const float focal_x, float focal_y,
							   const float kernel_size,
							   int *radii,
							   float2 *points_xy_image,
							   float *depths,
							   float *cov3Ds,
							   float *view2gaussians,
							   float *quadricCoeffs,
							   float *rgb,
							   float4 *conic_opacity,
							   const dim3 grid,
							   uint32_t *tiles_touched,
							   bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

	// access scale and rotation once to reduce IO
	const glm::vec3 scale = scales[idx];
	const glm::vec4 rot = rotations[idx];

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float *cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scale, rot, cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float4 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, kernel_size, cov3D, viewmatrix);
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]}; //* cov.w };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// view to gaussian coordinate system
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (view2gaussian_precomp == nullptr)
	{
		// printf("view2gaussian_precomp is nullptr\n");
		computeView2Gaussian(scale, p_orig, rot, viewmatrix, view2gaussians + idx * 16, quadricCoeffs + idx * 10);
	}
}

__global__ void computeRatioCUDA(
	float *dCount,
	const float *dReject,
	int P)
{
	// global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; idx < P; idx += stride)
	{
		float temp = dCount[idx];
		dCount[idx] = dReject[idx] / temp;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	gradientCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		float xg_thresh,
		float focal_x, float focal_y,
		const float *__restrict__ features,
		const float *__restrict__ view2gaussian,
		const float *__restrict__ quadricCoeffs,
		const glm::vec4 *__restrict__ rotations,
		const float3 *__restrict__ means3D,
		const float3 *__restrict__ scales,
		const float4 *__restrict__ conic_opacity,
		float *__restrict__ usedCount, float *__restrict__ rejectCount)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x + 0.5, (float)pix.y + 0.5}; // TODO plus 0.5

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// create the ray i.e. define the ray origin
	float2 ray = {(pixf.x - W / 2.) / focal_x, (pixf.y - H / 2.) / focal_y};

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	__shared__ float collected_quadricCoeffs[BLOCK_SIZE * 10]; // TODO we only need 12
	__shared__ float3 collected_scale[BLOCK_SIZE];
	// EV3DGS
	__shared__ float3 collected_means3d[BLOCK_SIZE];
	__shared__ glm::vec4 collected_rot[BLOCK_SIZE];
	// END EV3DGS

	// Initialize helper variables
	float T = 1.0f;

	// EV3DGS
	float alpha;
	float myT = 1.0f;
	float3 dC_dXg;
	float sum_xg[3] = {0.0f, 0.0f, 0.0f};
	float3 gradient;
	int index[NUM_CONTRIBUTORS] = {0};
	float max_power[NUM_CONTRIBUTORS] = {0.f};
	float xg[NUM_CONTRIBUTORS * 3] = {0.f};
	bool grad_score = false;
	int gIdReject[200];
	int gIdTotal[200];
	int countRejectIdx = 0;
	int countTotalIdx = 0;
	float sum_m[3] = {0.0f, 0.0f, 0.0f};
	float sum_s[3] = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < 200; i++)
	{
		gIdReject[i] = -1;
		gIdTotal[i] = -1;
	}

	// initialize index array in reverse order
	for (int i = 0; i < NUM_CONTRIBUTORS; i++)
	{
		index[i] = NUM_CONTRIBUTORS - 1 - i;
		max_power[i] = -FLT_MAX;
	}
	// END EV3DGS

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ii = 0; ii < 16; ii++)
				collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			// each record: 10 floats (40B)
			const float *__restrict__ base = &quadricCoeffs[coll_id * 10];
			float *__restrict__ out = &collected_quadricCoeffs[block.thread_rank() * 10];
			// Reinterpret as vectors of float2 (8-byte alignement; OK with 40B stride)
			const float2 *__restrict s2 = reinterpret_cast<const float2 *>(base);
			float2 *__restrict__ d2 = reinterpret_cast<float2 *>(out);
			// 0..1, 2..3, 4..5, 6..7, 8..9
			d2[0] = s2[0];
			d2[1] = s2[1];
			d2[2] = s2[2];
			d2[3] = s2[3];
			d2[4] = s2[4];

			collected_scale[block.thread_rank()] = scales[coll_id];
			// EV3DGS
			collected_means3d[block.thread_rank()] = means3D[coll_id];
			collected_rot[block.thread_rank()] = rotations[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float4 con_o = collected_conic_opacity[j];
			float *view2gaussian_j = collected_view2gaussian + j * 16;
			float *quadricCoeffs_j = collected_quadricCoeffs + j * 10;
			float3 scale_j = collected_scale[j];
			float3 ray_point = {ray.x, ray.y, 1.0};

			// EQ.2 from GOF paper - camera pos is at zero in view space/coordinate system
			float3 cam_pos_local = {view2gaussian_j[12], view2gaussian_j[13], view2gaussian_j[14]}; // translate camera center to gaussian's local coordinate system

			// EQ.3 from GOF paper
			float3 ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_j); // rotate ray to gaussian's local coordinate system

			// compute the minimal value
			const float normal[3] = {
				quadricCoeffs_j[0] * ray_point.x + quadricCoeffs_j[1] * ray_point.y + quadricCoeffs_j[2],
				quadricCoeffs_j[1] * ray_point.x + quadricCoeffs_j[3] * ray_point.y + quadricCoeffs_j[4],
				quadricCoeffs_j[2] * ray_point.x + quadricCoeffs_j[4] * ray_point.y + quadricCoeffs_j[5]};

			// use AA, BB, CC so that the name is unique
			float AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
			float BB = 2 * (quadricCoeffs_j[6] * ray_point.x + quadricCoeffs_j[7] * ray_point.y + quadricCoeffs_j[8]);
			float CC = quadricCoeffs_j[9];

			// t is the depth of the gaussian
			float t = -BB / (2 * AA);

			// my_t is not necessary because scale is cancelled out in the division
			// double myA = ray_local.x * ray_local.x + ray_local.y * ray_local.y + ray_local.z * ray_local.z;
			// double myB = 2 * (ray_local.x * cam_pos_local.x + ray_local.y * cam_pos_local.y + ray_local.z * cam_pos_local.z);
			// float my_t = -myB / (2 * myA);

			// depth must be positive otherwise it is not valid and we skip it
			float near_plane = NEAR_PLANE; //.75*(sqrt(viewmatrix[12]*viewmatrix[12]+viewmatrix[13]*viewmatrix[13]+viewmatrix[14]*viewmatrix[14])) + .01;
			if (t <= near_plane)
			{
				continue;
			}

			// EQ.4 from GOF paper - point of intersection in the local gaussian space
			float3 x_g = {
				cam_pos_local.x + t * ray_local.x, // if curious replace t with my_t
				cam_pos_local.y + t * ray_local.y,
				cam_pos_local.z + t * ray_local.z};

			double myMV = x_g.x * x_g.x + x_g.y * x_g.y + x_g.z * x_g.z;
			float myPow = -0.5f * myMV;
			if (myPow > 0.0f)
			{
				myPow = 0.0f;
			}
			float myAlpha = min(0.99f, con_o.w * exp(myPow));
			float myTestT = myT * (1 - myAlpha);

			double min_value = -(BB / AA) * (BB / 4.) + CC;

			float power = -0.5f * min_value;
			if (power > 0.0f)
			{
				power = 0.0f;
			}

			alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
			{
				continue;
			}

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float gColor = 1; // features[collected_id[j] * CHANNELS + i];
			float xg_first_term = gColor * myAlpha * myT;

			dC_dXg.x = xg_first_term * (sum_xg[0] - x_g.x);
			dC_dXg.y = xg_first_term * (sum_xg[1] - x_g.y);
			dC_dXg.z = xg_first_term * (sum_xg[2] - x_g.z);
			float gs_xg = dC_dXg.x * dC_dXg.x + dC_dXg.y * dC_dXg.y + dC_dXg.z * dC_dXg.z;

			if (countTotalIdx < 200)
			{
				gIdTotal[countTotalIdx] = collected_id[j];
				countTotalIdx += 1;
			}
			if (xg_thresh != 0.0 && countRejectIdx < 200 && gs_xg > xg_thresh)
			{
				gIdReject[countRejectIdx] = collected_id[j];
				countRejectIdx += 1;
				continue;
			}
			float denom = 1.f - alpha;
			sum_xg[0] += (myAlpha * x_g.x) / denom;
			sum_xg[1] += (myAlpha * x_g.y) / denom;
			sum_xg[2] += (myAlpha * x_g.z) / denom;

			T = test_T;
			myT = myTestT;
		}
	}
	if (inside)
	{
		for (int j = 0; j < 200; j++)
		{
			if (gIdReject[j] >= 0)
			{
				atomicAdd(&rejectCount[gIdReject[j]], 1.0);
			}
			if (gIdTotal[j] >= 0)
			{
				atomicAdd(&usedCount[gIdTotal[j]], 1.0);
			}
		}
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderRasterizationCUDA(
		const float *__restrict__ usedCount,
		const bool two_pass, float cc_thresh,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ features,
		const float4 *__restrict__ conic_opacity,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		float *__restrict__ out_color,
		const float *__restrict__ depths)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_reject_ratio[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_reject_ratio[block.thread_rank()] = usedCount[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

			// EV3DGS - skip if in two-pass mode and reject ratio too high
			if (two_pass && collected_reject_ratio[j] >= cc_thresh)
			{
				continue;
			}
			// END EV3DGS

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderRayMarchCUDA(
		const float *__restrict__ usedCount,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const bool two_pass, float cc_thresh,
		float focal_x, float focal_y,
		const float *__restrict__ features,
		const float *__restrict__ quadricCoeffs,
		const glm::vec4 *__restrict__ rotations,
		const float3 *__restrict__ means3D,
		const float3 *__restrict__ scales,
		const float4 *__restrict__ conic_opacity,
		const float *__restrict__ bg_color,
		float *__restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;

	float2 pixf = {(float)pix.x + 0.5, (float)pix.y + 0.5}; // TODO plus 0.5

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// create the ray i.e. define the ray origin
	float2 ray = {(pixf.x - W / 2.) / focal_x, (pixf.y - H / 2.) / focal_y};

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_quadricCoeffs[BLOCK_SIZE * 10];
	__shared__ float3 collected_scale[BLOCK_SIZE];
	// EV3DGS
	__shared__ float3 collected_means3d[BLOCK_SIZE];
	__shared__ glm::vec4 collected_rot[BLOCK_SIZE];
	__shared__ float collected_reject_ratio[BLOCK_SIZE];
	// END EV3DGS

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	int max_contributor = -1;
	float C[OUTPUT_CHANNELS - 1] = {0}; // NOT SAVING DISTORTION IN BUFFER

	float alpha;

	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			// each record: 10 floats (40B)
			const float *__restrict__ base = &quadricCoeffs[coll_id * 10];
			float *__restrict__ out = &collected_quadricCoeffs[block.thread_rank() * 10];
			// Reinterpret as vectors of float2 (8-byte alignement; OK with 40B stride)
			const float2 *__restrict s2 = reinterpret_cast<const float2 *>(base);
			float2 *__restrict__ d2 = reinterpret_cast<float2 *>(out);
			// 0..1, 2..3, 4..5, 6..7, 8..9
			d2[0] = s2[0];
			d2[1] = s2[1];
			d2[2] = s2[2];
			d2[3] = s2[3];
			d2[4] = s2[4];

			collected_scale[block.thread_rank()] = scales[coll_id];
			// EV3DGS
			collected_means3d[block.thread_rank()] = means3D[coll_id];
			collected_rot[block.thread_rank()] = rotations[coll_id];
			collected_reject_ratio[block.thread_rank()] = usedCount[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (two_pass && collected_reject_ratio[j] >= cc_thresh)
			{
				continue;
			}

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float4 con_o = collected_conic_opacity[j];
			float *quadricCoeffs_j = collected_quadricCoeffs + j * 10;

			float3 scale_j = collected_scale[j];
			float3 ray_point = {ray.x, ray.y, 1.0};

			const float normal[3] = {
				quadricCoeffs_j[0] * ray_point.x + quadricCoeffs_j[1] * ray_point.y + quadricCoeffs_j[2],
				quadricCoeffs_j[1] * ray_point.x + quadricCoeffs_j[3] * ray_point.y + quadricCoeffs_j[4],
				quadricCoeffs_j[2] * ray_point.x + quadricCoeffs_j[4] * ray_point.y + quadricCoeffs_j[5]};

			// use AA, BB, CC so that the name is unique
			float AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
			float BB = 2 * (quadricCoeffs_j[6] * ray_point.x + quadricCoeffs_j[7] * ray_point.y + quadricCoeffs_j[8]);
			float CC = quadricCoeffs_j[9];

			// t is the depth of the gaussian
			float t = -BB / (2 * AA);

			float near_plane = NEAR_PLANE; //.75*(sqrt(viewmatrix[12]*viewmatrix[12]+viewmatrix[13]*viewmatrix[13]+viewmatrix[14]*viewmatrix[14])) + .01;
			if (t <= near_plane)
			{
				continue;
			}

			const float scale = 1.0f / sqrt(AA + 1e-7);
			// the scale of the gaussian is 1.f / sqrt(AA)
			double min_value = -(BB / AA) * (BB / 4.) + CC; // working alts:  AA*t*t + BB*t + CC; (when x_g scaled) x_g.x*x_g.x + x_g.y*x_g.y + x_g.z*x_g.z;

			float power = -0.5f * min_value;
			if (power > 0.0f)
			{
				power = 0.0f;
			}

			// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
			const float max_t = t;
			const float mapped_max_t = (FAR_PLANE * max_t - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * max_t);

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
			{
				continue;
			}

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += (features[collected_id[j] * CHANNELS + ch] * alpha * T);

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// rendered RGB - NOTE - ch=RGB
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

__global__ void rasterizeForwardCUDA(
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
	const float3 &__restrict__ background)
{
	// each thread draws one pixel, but also timeshares caching gaussians in a
	// shared tile

	auto block = cg::this_thread_block();
	int32_t tile_id =
		block.group_index().y * tile_bounds.x + block.group_index().x;
	unsigned i =
		block.group_index().y * block.group_dim().y + block.thread_index().y;
	unsigned j =
		block.group_index().x * block.group_dim().x + block.thread_index().x;

	float px = (float)j + 0.5;
	float py = (float)i + 0.5;
	int32_t pix_id = i * img_size.x + j;

	// return if out of bounds
	// keep not rasterizing threads around for reading data
	bool inside = (i < img_size.y && j < img_size.x);
	bool done = !inside;

	// have all threads in tile process the same gaussians in batches
	// first collect gaussians between range.x and range.y in batches
	// which gaussians to look through in this tile
	int2 range = tile_bins[tile_id];
	const int block_size = block.size();
	int num_batches = (range.y - range.x + block_size - 1) / block_size;

	__shared__ int32_t id_batch[BLOCK_SIZE];
	__shared__ float3 xy_opacity_batch[BLOCK_SIZE];
	__shared__ float3 conic_batch[BLOCK_SIZE];

	// current visibility left to render
	float T = 1.f;

	// index of most recent gaussian to write to this thread's pixel
	int cur_idx = 0;

	// collect and process batches of gaussians
	// each thread loads one gaussian at a time before rasterizing its
	// designated pixel
	int tr = block.thread_rank();
	float3 pix_out = {0.f, 0.f, 0.f};
	for (int b = 0; b < num_batches; ++b)
	{
		// resync all threads before beginning next batch
		// end early if entire tile is done
		if (__syncthreads_count(done) >= block_size)
		{
			break;
		}

		// each thread fetch 1 gaussian from front to back
		// index of gaussian to load
		int batch_start = range.x + block_size * b;
		int idx = batch_start + tr;
		if (idx < range.y)
		{
			int32_t g_id = gaussian_ids_sorted[idx];
			id_batch[tr] = g_id;
			const float2 xy = xys[g_id];
			const float opac = opacities[g_id];
			xy_opacity_batch[tr] = {xy.x, xy.y, opac};
			conic_batch[tr] = conics[g_id];
		}

		// wait for other threads to collect the gaussians in batch
		block.sync();

		// process gaussians in the current batch for this pixel
		int batch_size = min(block_size, range.y - batch_start);
		for (int t = 0; (t < batch_size) && !done; ++t)
		{
			const float3 conic = conic_batch[t];
			const float3 xy_opac = xy_opacity_batch[t];
			const float opac = xy_opac.z;
			const float2 delta = {xy_opac.x - px, xy_opac.y - py};
			const float sigma = 0.5f * (conic.x * delta.x * delta.x +
										conic.z * delta.y * delta.y) +
								conic.y * delta.x * delta.y;

			const float alpha = min(0.999f, opac * __expf(-sigma));
			if (sigma < 0.f || alpha < 1.f / 255.f)
			{
				continue;
			}

			const float next_T = T * (1.f - alpha);
			if (next_T <= 1e-4f)
			{ // this pixel is done
				// we want to render the last gaussian that contributes and note
				// that here idx > range.x so we don't underflow
				done = true;
				break;
			}

			int32_t g = id_batch[t];
			const float vis = alpha * T;

			// float area = 3.14159 * sqrt(conic.x)*sqrt(conic.z);

			cur_idx = batch_start + t;

			const float3 c = colors[g];
			pix_out.x = pix_out.x + c.x * vis;
			pix_out.y = pix_out.y + c.y * vis;
			pix_out.z = pix_out.z + c.z * vis;

			T = next_T;
		}
	}

	if (inside)
	{
		// add background
		final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
		final_index[pix_id] =
			cur_idx; // index of in bin of last gaussian in this pixel
		float3 final_color;
		final_color.x = pix_out.x + T * background.x;
		final_color.y = pix_out.y + T * background.y;
		final_color.z = pix_out.z + T * background.z;
		out_img[pix_id] = final_color;
	}
}

void FORWARD::rasterizeForward(
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
	const float3 &__restrict__ background)
{

	rasterizeForwardCUDA<<<tile_bounds, block>>>(
		tile_bounds,
		img_size,
		gaussian_ids_sorted,
		tile_bins,
		xys,
		conics,
		colors,
		opacities,
		final_Ts,
		final_index,
		out_img,
		background);
}

void FORWARD::computeRatio(
	int blocks, int threads,
	float *dCount,
	const float *dReject,
	int P)
{
	computeRatioCUDA<<<blocks, threads>>>(
		dCount,
		dReject,
		P);
}

void FORWARD::gradient(
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	float xg_thresh,
	float focal_x, float focal_y,
	const float *colors,
	const float *view2gaussian,
	const float *quadricCoeffs,
	const glm::vec4 *rotations,
	const float3 *means3D,
	const float3 *scales,
	const float4 *conic_opacity,
	float *usedCount, float *rejectCount)
{
	gradientCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		xg_thresh,
		focal_x, focal_y,
		colors,
		view2gaussian,
		quadricCoeffs,
		rotations,
		means3D,
		scales,
		conic_opacity,
		usedCount, rejectCount);
}

void FORWARD::renderRasterization(
	const float *usable_g,
	const bool two_pass, float cc_thresh,
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float2 *means2D,
	const float *colors,
	const float4 *conic_opacity,
	float *final_T,
	uint32_t *n_contrib,
	const float *bg_color,
	float *out_color,
	float *depths)
{
	renderRasterizationCUDA<NUM_CHANNELS><<<grid, block>>>(
		usable_g,
		two_pass, cc_thresh,
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths);
}

void FORWARD::renderRayMarch(
	const float *usable_g,
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const bool two_pass, float cc_thresh,
	float focal_x, float focal_y,
	const float *colors,
	const float *quadricCoeffs,
	const glm::vec4 *rotations,
	const float3 *means3D,
	const float3 *scales,
	const float4 *conic_opacity,
	const float *bg_color,
	float *out_color)
{
	renderRayMarchCUDA<NUM_CHANNELS><<<grid, block>>>(
		// reinterpret_cast<const bool*>(usable_g),
		usable_g,
		ranges,
		point_list,
		W, H,
		two_pass, cc_thresh,
		focal_x, focal_y,
		colors,
		quadricCoeffs,
		rotations,
		means3D,
		scales,
		conic_opacity,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
						 const float *means3D,
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
						 float2 *means2D,
						 float *depths,
						 float *cov3Ds,
						 float *view2gaussians,
						 float *quadricCoeffs,
						 float *rgb,
						 float4 *conic_opacity,
						 const dim3 grid,
						 uint32_t *tiles_touched,
						 bool prefiltered)
{
#define COMMA ,
	CHECK_CUDA(preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256 COMMA 256>>>(
				   P, D, M,
				   means3D,
				   scales,
				   rotations,
				   opacities,
				   shs,
				   clamped,
				   cov3D_precomp,
				   colors_precomp,
				   view2gaussian_precomp,
				   viewmatrix,
				   projmatrix,
				   cam_pos,
				   W, H,
				   tan_fovx, tan_fovy,
				   focal_x, focal_y,
				   kernel_size,
				   radii,
				   means2D,
				   depths,
				   cov3Ds,
				   view2gaussians,
				   quadricCoeffs,
				   rgb,
				   conic_opacity,
				   grid,
				   tiles_touched,
				   prefiltered),
			   true)
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessPointsCUDA(int P, int D, int M,
									 const float *points3D,
									 const float *viewmatrix,
									 const float *projmatrix,
									 const glm::vec3 *cam_pos,
									 const int W, int H,
									 const float tan_fovx, float tan_fovy,
									 const float focal_x, float focal_y,
									 float2 *points2D,
									 float *depths,
									 const dim3 grid,
									 uint32_t *tiles_touched,
									 bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, points3D, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = {points3D[3 * idx], points3D[3 * idx + 1], points3D[3 * idx + 2]};
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

	float2 point_image = {focal_x * p_view.x / (p_view.z + 0.0000001f) + W / 2., focal_y * p_view.y / (p_view.z + 0.0000001f) + H / 2.};

	// If the point is outside the image, quit.
	if (point_image.x < 0 || point_image.x >= W || point_image.y < 0 || point_image.y >= H)
		return;

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	points2D[idx] = point_image;
	tiles_touched[idx] = 1;
}

void FORWARD::preprocess_points(int PN, int D, int M,
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
								bool prefiltered)
{
	preprocessPointsCUDA<NUM_CHANNELS><<<(PN + 255) / 256, 256>>>(
		PN, D, M,
		points3D,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points2D,
		depths,
		grid,
		tiles_touched,
		prefiltered);
}