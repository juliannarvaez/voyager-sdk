// Copyright Axelera AI, 2025
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"

#include <optional>

#include <opencv2/core/ocl.hpp>

namespace
{
struct padding_properties {
  std::vector<std::vector<int>> paddings;
  std::optional<int8_t> fill{};
  std::vector<int> in_shape{};
  std::vector<int> out_shape{};
};


} // namespace

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "padding",
    "fill", "input_shape", "output_shape" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<padding_properties> prop = std::make_shared<padding_properties>();
  prop->paddings = Ax::get_property(
      input, "padding", "padding_properties", std::vector<std::vector<int>>{});
  // For backward compatibility - if no paddings were specified but there's a single padding vector, convert it
  if (prop->paddings.empty()) {
    std::vector<int> single_padding = Ax::get_property(
        input, "padding", "padding_properties", std::vector<int>{});
    if (!single_padding.empty()) {
      prop->paddings.push_back(single_padding);
    }
  }
  prop->fill = Ax::get_property(input, "fill", "padding_properties", prop->fill);
  prop->in_shape
      = Ax::get_property(input, "input_shape", "padding_properties", prop->in_shape);
  prop->out_shape
      = Ax::get_property(input, "output_shape", "padding_properties", prop->out_shape);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const padding_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_padding requires tensor input");
  }
  auto input = std::get<AxTensorsInterface>(interface);


  // Make sure we have at least one tensor
  if (input.empty() || input[0].bytes != 1) {
    throw std::runtime_error("transform_padding requires at least one int8 tensor input");
  }

  // Make sure we have paddings for each tensor or at least one default padding
  if (prop->paddings.empty()) {
    throw std::runtime_error("transform_padding: no padding configurations provided");
  }

  if (prop->paddings.size() < input.size()) {
    throw std::runtime_error("transform_padding: fewer padding configurations than tensors, expected "
                             + std::to_string(input.size()) + " but got "
                             + std::to_string(prop->paddings.size()));
  }

  // Validate each padding configuration
  for (size_t i = 0; i < std::min(prop->paddings.size(), input.size()); ++i) {
    const auto &padding = prop->paddings[i];
    if ((padding.size() % 2) != 0) {
      throw std::runtime_error("transform_padding: padding must be a multiple of 2:"
                               + ax_utils::sizes_to_string(padding));
    }
    if (padding.size() / 2 > input[i].sizes.size()) {
      throw std::runtime_error("transform_padding: padding "
                               + ax_utils::sizes_to_string(padding) + " too long for input tensor "
                               + ax_utils::sizes_to_string(input[i].sizes));
    }
  }

  if (!ax_utils::validate_shape(prop->in_shape, input[0].sizes)) {
    throw std::runtime_error("transform_padding: input_shape "
                             + ax_utils::sizes_to_string(prop->in_shape) + " does not match input tensor "
                             + ax_utils::sizes_to_string(input[0].sizes));
  }

  auto output = input;

  // Calculate output sizes for each tensor based on its padding
  for (size_t i = 0; i < input.size(); ++i) {
    // Use the appropriate padding for this tensor (or the last one if we have fewer paddings than tensors)
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_sizes = prop->in_shape.empty() ? input[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_sizes, padding);

    if (i == 0 && !ax_utils::validate_shape(prop->out_shape, info.out_sizes)) {
      throw std::runtime_error("transform_padding: output_shape "
                               + ax_utils::sizes_to_string(prop->out_shape) + " does not match calculated output tensor "
                               + ax_utils::sizes_to_string(info.out_sizes));
    }

    output[i].sizes = prop->out_shape.empty() ? info.out_sizes : prop->out_shape;
  }


  return { output };
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const padding_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto input_tensors = std::get<AxTensorsInterface>(input);
  auto output_tensors = std::get<AxTensorsInterface>(output);

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_shape = prop->in_shape.empty() ? input_tensors[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_shape, padding);

    auto *src = static_cast<const uint8_t *>(input_tensors[i].data);
    auto *dst = static_cast<uint8_t *>(output_tensors[i].data);

    // Fast path for common NHWC padding: [N,H,W,C] with zero padding on C.
    // For padding like 0,0,1,1,1,15,0,0 on [1,640,640,3]:
    //   - C (last dim) has no padding → each row of W*C bytes is contiguous
    //   - We copy H rows of W*C bytes each, with proper stride in the output
    //   - Only need H*W row copies (640 rows) instead of H*W*C pixel copies (409600)
    const auto ndims = info.in_sizes.size();
    if (!info.is_crop && ndims >= 3 && prop->fill) {
      // Check that the last dimension has zero padding
      const int pad_last_lo = padding[2 * (ndims - 1)];
      const int pad_last_hi = padding[2 * (ndims - 1) + 1];

      if (pad_last_lo == 0 && pad_last_hi == 0) {
        // Compute the innermost contiguous row size (all trailing dims with zero padding)
        // For NHWC with C-zero-pad and W-nonzero-pad: row = C (too small)
        // But we can still treat W*C as a "logical row" and handle W-padding at the row level.

        // Strategy: iterate over all dimensions except the last two (W, C).
        // For each "H-row", copy W_in * C bytes from src to the correct offset in dst.
        // The output row stride is W_out * C bytes.

        // Find the two innermost dimensions: second-to-last = W, last = C
        // More generally, find the innermost dim WITH padding (call it the "padded dim")
        // and treat everything below it as the contiguous chunk.

        // Compute sizes
        const size_t C = info.in_sizes[ndims - 1]; // last dim, no padding
        const size_t W_in = info.in_sizes[ndims - 2];
        const size_t W_out = info.out_sizes[ndims - 2];
        const int W_pad_lo = padding[2 * (ndims - 2)];

        const size_t row_bytes_in = W_in * C;
        const size_t row_bytes_out = W_out * C;

        // Total number of "H-rows": product of all dims except the last two
        size_t total_h_rows_in = 1;
        size_t total_h_rows_out = 1;
        for (size_t d = 0; d < ndims - 2; ++d) {
          total_h_rows_in *= info.in_sizes[d];
          total_h_rows_out *= info.out_sizes[d];
        }

        // If the dims above W also have padding, we need to handle the offset.
        // For the typical case [N=1, H, W, C] with padding [0,0, H_lo,H_hi, W_lo,W_hi, 0,0]:
        // - total_h_rows_in = N * H_in = H_in
        // - total_h_rows_out = N * H_out = H_in + H_lo + H_hi
        // - H_pad_lo rows at the top are fill-only
        // - Then H_in rows have data (offset by W_pad_lo * C in each row)
        // - Then H_pad_hi rows at the bottom are fill-only

        // For the general N-d case, we can compute the linear offset of each
        // input H-row in the output buffer. But for the common 4D NHWC case,
        // this simplifies to a contiguous block of rows with a fixed offset.

        // Simple, fast approach: memset the entire output, then copy rows.
        const uint8_t fill_val = static_cast<uint8_t>(*prop->fill);
        size_t total_out_bytes = 1;
        for (size_t d = 0; d < ndims; ++d) total_out_bytes *= info.out_sizes[d];
        std::memset(dst, fill_val, total_out_bytes);

        // Now compute the starting output offset for the first input row.
        // For each dim d < ndims-2, the offset contribution is padding[2*d] * (product of out_dims after d).
        size_t base_dst_offset = W_pad_lo * C; // W-padding offset within each row
        {
          size_t outer_stride = row_bytes_out; // stride for dim ndims-3
          for (int d = static_cast<int>(ndims) - 3; d >= 0; --d) {
            base_dst_offset += padding[2 * d] * outer_stride;
            outer_stride *= info.out_sizes[d];
          }
        }

        // Check if all dims above W have contiguous (zero) padding
        // If so, input rows map to a contiguous block in output (just offset).
        bool outer_contiguous = true;
        for (size_t d = 0; d < ndims - 2; ++d) {
          if (padding[2 * d] != 0 || padding[2 * d + 1] != 0) {
            // There IS padding on an outer dim, but if all outer dims except one
            // have zero padding, we can still do a contiguous block.
            // For simplicity, handle the case where at most the first non-trivial
            // outer dim has padding (typical for [N,H,W,C]).
            outer_contiguous = false;
            break;
          }
        }

        if (outer_contiguous || ndims == 3) {
          // All outer dims have zero padding. Input rows are a contiguous block
          // in the output, starting at base_dst_offset with stride row_bytes_out.
          // Since outer padding is zero, total_h_rows_in == total_h_rows_out,
          // but we still need to account for W padding per row.
          for (size_t r = 0; r < total_h_rows_in; ++r) {
            std::memcpy(dst + base_dst_offset + r * row_bytes_out,
                        src + r * row_bytes_in,
                        row_bytes_in);
          }
          continue;
        }

        // Common 4D case: [N, H, W, C] with H and/or W padding.
        // N is typically 1, but handle general N.
        if (ndims == 4) {
          const size_t N_in = info.in_sizes[0];
          const size_t H_in = info.in_sizes[1];
          const size_t H_out = info.out_sizes[1];
          const int H_pad_lo = padding[2]; // padding on H dimension
          const size_t slice_out = H_out * row_bytes_out;
          const size_t slice_in = H_in * row_bytes_in;
          const int N_pad_lo = padding[0];

          for (size_t n = 0; n < N_in; ++n) {
            const size_t n_dst = (n + N_pad_lo) * slice_out;
            const size_t n_src = n * slice_in;
            for (size_t h = 0; h < H_in; ++h) {
              std::memcpy(
                dst + n_dst + (h + H_pad_lo) * row_bytes_out + W_pad_lo * C,
                src + n_src + h * row_bytes_in,
                row_bytes_in);
            }
          }
          continue;
        }
      }
    }

    // Fallback: use OpenCV N-d copy (for complex padding cases)
    cv::ocl::setUseOpenCL(false);
    cv::Mat input_mat(info.in_sizes, CV_8UC1, input_tensors[i].data);
    cv::Mat output_mat(info.out_sizes, CV_8UC1, output_tensors[i].data);

    if (info.is_crop) {
      input_mat(info.ranges).copyTo(output_mat);
    } else {
      if (prop->fill) {
        output_mat.setTo(*prop->fill);
      }
      input_mat.copyTo(output_mat(info.ranges));
    }
  }
}
