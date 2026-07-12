#include "dg_capi.h"

#include <stddef.h>
#include <stdint.h>

int main(void) {
  const char *options = "{\"shape\":[1,4],\"echo_inputs\":true}";
  struct DgBackend *backend = NULL;
  struct DgTensor *input = NULL;
  struct DgTensor *output = NULL;
  const uint8_t *output_data = NULL;
  size_t output_length = 0;
  size_t shape[] = {1, 4};
  float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  size_t input_count = 0;
  size_t output_count = 0;
  struct DgBackendCapabilities capabilities;

  if (dg_backend_create(Mock, (const uint8_t *)"", 0, options, &backend) != Ok ||
      dg_backend_io_counts(backend, &input_count, &output_count) != Ok ||
      dg_backend_capabilities(backend, &capabilities) != Ok ||
      dg_tensor_create((const uint8_t *)values, sizeof(values), shape, 2, F32, Nc, Cpu,
                       &input) != Ok) {
    dg_backend_free(backend);
    dg_tensor_free(input);
    return 1;
  }

  const struct DgTensor *inputs[] = {input};
  if (dg_backend_run(backend, inputs, 1, &output, 1, &output_count) != Ok ||
      dg_tensor_data(output, &output_data, &output_length) != Ok) {
    dg_tensor_free(input);
    dg_tensor_free(output);
    dg_backend_free(backend);
    return 1;
  }

  (void)output_data;
  (void)output_length;
  dg_tensor_free(output);
  dg_tensor_free(input);
  dg_backend_free(backend);
  return 0;
}
