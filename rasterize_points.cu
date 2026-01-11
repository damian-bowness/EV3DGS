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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_ev3dgs/config.h"
#include "cuda_ev3dgs/auxiliary.h"
#include "cuda_ev3dgs/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <cuda_runtime.h>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<torch::Tensor>
RenderGaussiansCUDA(
	const torch::Tensor &background,
	const bool rastFlag,
	const bool two_pass,
	const float xg_thresh,
	const float cc_thresh,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const torch::Tensor &gaussian_index,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &view2gaussian_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float kernel_size,
	const torch::Tensor &subpixel_offset,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	auto bool_opts = means3D.options().dtype(torch::kBool);

	torch::Tensor out_color = torch::full({3 * H * W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			rastFlag,
			two_pass, xg_thresh, cc_thresh,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			gaussian_index.contiguous().data<int>(),
			cov3D_precomp.contiguous().data<float>(),
			view2gaussian_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			kernel_size,
			subpixel_offset.contiguous().data<float>(),
			prefiltered,
			out_color.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
	}
	return out_color;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeForwardCUDA(
	const std::tuple<int, int, int> tile_bounds,
	const std::tuple<int, int, int> block,
	const std::tuple<int, int, int> img_size,
	const torch::Tensor &gaussian_ids_sorted,
	const torch::Tensor &tile_bins,
	const torch::Tensor &xys,
	const torch::Tensor &conics,
	const torch::Tensor &colors,
	const torch::Tensor &opacities,
	const torch::Tensor &background)
{

	dim3 tile_bounds_dim3;
	tile_bounds_dim3.x = std::get<0>(tile_bounds);
	tile_bounds_dim3.y = std::get<1>(tile_bounds);
	tile_bounds_dim3.z = std::get<2>(tile_bounds);

	dim3 block_dim3;
	block_dim3.x = std::get<0>(block);
	block_dim3.y = std::get<1>(block);
	block_dim3.z = std::get<2>(block);

	dim3 img_size_dim3;
	img_size_dim3.x = std::get<0>(img_size);
	img_size_dim3.y = std::get<1>(img_size);
	img_size_dim3.z = std::get<2>(img_size);

	const int channels = colors.size(1);
	const int img_width = img_size_dim3.x;
	const int img_height = img_size_dim3.y;

	torch::Tensor out_img = torch::zeros(
		{img_height, img_width, channels}, xys.options().dtype(torch::kFloat32));
	torch::Tensor final_Ts = torch::zeros(
		{img_height, img_width}, xys.options().dtype(torch::kFloat32));
	torch::Tensor final_idx = torch::zeros(
		{img_height, img_width}, xys.options().dtype(torch::kInt32));

	CudaRasterizer::Rasterizer::rasterizeForward(
		block_dim3,
		tile_bounds_dim3,
		img_size_dim3,
		gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
		(int2 *)tile_bins.contiguous().data_ptr<int>(),
		(float2 *)xys.contiguous().data_ptr<float>(),
		(float3 *)conics.contiguous().data_ptr<float>(),
		(float3 *)colors.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		final_Ts.contiguous().data_ptr<float>(),
		final_idx.contiguous().data_ptr<int>(),
		(float3 *)out_img.contiguous().data_ptr<float>(),
		*(float3 *)background.contiguous().data_ptr<float>());

	return std::make_tuple(out_img, final_Ts, final_idx);
}