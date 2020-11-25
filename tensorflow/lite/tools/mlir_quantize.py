# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Run the MLIR quantizer on a calibrated TFLite model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow.lite.python.convert import mlir_quantize

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tflite_file', None,
                    'Full path name to the input tflite file.')
flags.DEFINE_string('output_tflite_file', None,
                    'Full path name to the output randomized tflite file.')
flags.DEFINE_boolean('disable_per_channel', False,
                     'To disable per channel '
                     '(i.e, enable per tensor) quantization')
flags.DEFINE_boolean('fully_quantize', False,
                     'To quantize the input and output to int8 as well')

flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')


def main(_):
  # Read the model
  with open(FLAGS.input_tflite_file, 'rb') as f:
    model = f.read()

  # Invoke MLIR quantizer
  quant_model = mlir_quantize(
      model, FLAGS.disable_per_channel, FLAGS.fully_quantize)

  # Write the model
  with open(FLAGS.output_tflite_file, 'wb') as f:
    f.write(quant_model)


if __name__ == '__main__':
  app.run(main)
