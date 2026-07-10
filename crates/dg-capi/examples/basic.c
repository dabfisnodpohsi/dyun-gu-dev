#include "dg_capi.h"

#include <stddef.h>
#include <stdint.h>

int main(void) {
  struct DgEngine *engine = NULL;
  const char *spec =
      "apiVersion: dg/v1\n"
      "kind: Graph\n"
      "nodes:\n"
      "  - name: input\n"
      "    kind: input\n"
      "    params: {}\n"
      "  - name: infer\n"
      "    kind: mock_inference\n"
      "    params: {shape: [1, 4], echo_inputs: true}\n"
      "  - name: sink\n"
      "    kind: sink\n"
      "    params: {}\n"
      "connections:\n"
      "  - input.out -> infer.in\n"
      "  - infer.out -> sink.in\n";
  size_t shape[] = {1, 4};
  float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  struct DgTensor *tensor = NULL;
  struct DgTensor *output = NULL;
  const uint8_t *output_data = NULL;
  size_t output_length = 0;
  size_t added_nodes = 0;
  size_t removed_nodes = 0;
  size_t updated_nodes = 0;
  size_t added_connections = 0;
  size_t removed_connections = 0;

  if (dg_engine_create(&engine) != Ok ||
      dg_engine_load_string(engine, Yaml, spec) != Ok ||
      dg_engine_diff_string(engine, Yaml, spec, &added_nodes, &removed_nodes,
                            &updated_nodes, &added_connections,
                            &removed_connections) != Ok ||
      dg_engine_build(engine) != Ok ||
      dg_tensor_create((const uint8_t *)input, sizeof(input), shape, 2, F32, Nc, Cpu, &tensor) != Ok ||
      dg_engine_push(engine, tensor) != Ok ||
      dg_engine_run(engine) != Ok ||
      dg_engine_poll(engine, &output) != Ok ||
      dg_tensor_data(output, &output_data, &output_length) != Ok) {
    dg_engine_free(engine);
    dg_tensor_free(tensor);
    dg_tensor_free(output);
    return 1;
  }

  (void)output_data;
  (void)output_length;
  dg_tensor_free(output);
  dg_tensor_free(tensor);
  dg_engine_free(engine);
  return 0;
}
