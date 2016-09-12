#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rnn_dropout_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void RNNDropoutForward(const int n, const Dtype* in,
    const unsigned int rand, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (rand > threshold) * scale;
  }
}

template <typename Dtype>
void RNNDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
  return;


//  const Dtype* bottom_data = bottom[0]->gpu_data();
//  Dtype* top_data = top[0]->mutable_gpu_data();
//  const int count = bottom[0]->count();
//  if (this->phase_ == TRAIN) {
//    caffe_gpu_rng_uniform(1, rand_);
//    // set thresholds
//    // NOLINT_NEXT_LINE(whitespace/operators)
//    RNNDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//        count, bottom_data, rand_, uint_thres_, scale_, top_data);
//    CUDA_POST_KERNEL_CHECK;
//  } else {
//    caffe_copy(count, bottom_data, top_data);
//  }
}

template <typename Dtype>
__global__ void RNNDropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int rand, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (rand > threshold);
  }
}

template <typename Dtype>
void RNNDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
  return;


//  if (propagate_down[0]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//    if (this->phase_ == TRAIN) {
//      const int count = bottom[0]->count();
//      // NOLINT_NEXT_LINE(whitespace/operators)
//      RNNDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
//        CAFFE_CUDA_NUM_THREADS>>>(
//          count, top_diff, rand_, uint_thres_, scale_, bottom_diff);
//      CUDA_POST_KERNEL_CHECK;
//    } else {
//      caffe_copy(top[0]->count(), top_diff, bottom_diff);
//    }
//  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RNNDropoutLayer);


}  // namespace caffe
