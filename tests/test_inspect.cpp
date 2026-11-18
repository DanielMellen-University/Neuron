#define NEURON_TEST_MAIN
#include "neuron/inspect.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <string>

namespace {
fixture::Bytes sample() {
  fixture::Bytes b;
  b.header(2, 6);
  b.text("general.architecture", "llama");
  b.text("general.name", "say \"hi\"\n\x01");
  b.uint("llama.context_length", 64);
  b.key("llama.rope.freq_base", 6); b.put(10000.5f);
  b.key("tokenizer.ggml.add_bos_token", 7); b.put<std::uint8_t>(1);
  b.key("tokenizer.ggml.token_type", 9); b.put<std::uint32_t>(5); b.put<std::uint64_t>(10);
  for (std::int32_t i = 0; i < 10; ++i) b.put<std::int32_t>(-i);
  b.tensor("token_embd.weight", {4, 3}, 0, 0);
  b.tensor("output_norm.weight", {4}, 1, 64);
  b.align();
  for (int i = 0; i < 12; ++i) b.put(0.0f);
  b.align();
  for (int i = 0; i < 4; ++i) b.put<std::uint16_t>(0);
  return b;
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
}
NEURON_TEST(header_metadata_and_tensors) {
  fixture::File f(sample());
  const neuron::Gguf file(f.path);
  const auto text = neuron::describe_gguf(file);
  CHECK(text.starts_with("GGUF v3, 6 metadata keys, 2 tensors, alignment 32, data at byte "));
  CHECK(has(text, "general.architecture         string \"llama\"\n"));
  CHECK(has(text, "\"say \\\"hi\\\"\\n\\x01\""));
  CHECK(has(text, "llama.context_length         u32    64\n"));
  CHECK(has(text, "f32    10000.5\n"));
  CHECK(has(text, "bool   true\n"));
  CHECK(has(text, "[i32 x 10] 0, -1, -2, -3, -4, -5, -6, -7, ... 2 more\n"));
  CHECK(has(text, "  token_embd.weight  F32  4 x 3 48 bytes at 0\n"));
  CHECK(has(text, "  output_norm.weight F16  4     8 bytes at 64\n"));
  // 16 weights in 56 bytes is 28 bits per weight.
  CHECK(has(text, "\n16 parameters in 56 B (28.00 bits per weight)\n"));
  CHECK(has(text, "\n  F32  1 tensors, 12 parameters, 48 B\n  F16  1 tensors, 4 parameters, 8 B\n"));
}
NEURON_TEST(json_output) {
  fixture::File f(sample());
  const neuron::Gguf file(f.path);
  neuron::InspectOptions options;
  options.array_items = 3;
  const auto text = neuron::describe_gguf_json(file, options);
  CHECK(text.starts_with("{\"version\":3,\"alignment\":32,\"data_offset\":"));
  CHECK(has(text, ",\"parameters\":16,\"tensor_bytes\":56,"));
  CHECK(has(text, "\"general.name\":{\"type\":\"string\",\"value\":\"say \\\"hi\\\"\\u000a\\u0001\"}"));
  CHECK(has(text, "\"llama.rope.freq_base\":{\"type\":\"f32\",\"value\":10000.5}"));
  CHECK(has(text, "\"tokenizer.ggml.token_type\":{\"type\":\"array\",\"value\":{\"element_type\":\"i32\",\"count\":10,\"items\":[0,-1,-2]}}"));
  CHECK(has(text, "{\"name\":\"output_norm.weight\",\"type\":\"F16\",\"shape\":[4],\"parameters\":4,\"bytes\":8,\"offset\":64}"));
  options.tensors = false; options.array_items = 0;
  const auto all = neuron::describe_gguf_json(file, options);
  CHECK(!has(all, "\"tensors\""));
  CHECK(has(all, "\"items\":[0,-1,-2,-3,-4,-5,-6,-7,-8,-9]"));
  // Invalid UTF-8 in a string is replaced so the output stays valid JSON.
  fixture::Bytes b; b.header(0, 1); b.text("general.name", "a\xff");
  fixture::File g(b);
  CHECK(has(neuron::describe_gguf_json(neuron::Gguf(g.path)), "\"value\":\"a\xef\xbf\xbd\""));
}
NEURON_TEST(summary_scales_units_and_handles_no_tensors) {
  fixture::Bytes b;
  b.header(1, 0);
  b.tensor("big", {1024, 1024}, 0, 0);
  b.align();
  b.data.resize(b.data.size() + 4u * 1024 * 1024);
  fixture::File f(b);
  CHECK_EQ(neuron::summarize_tensors(neuron::Gguf(f.path)),
           std::string("1048576 parameters in 4.0 MiB (32.00 bits per weight)\n  F32  1 tensors, 1048576 parameters, 4.0 MiB\n"));
  fixture::Bytes empty; empty.header(0, 0);
  fixture::File g(empty);
  CHECK_EQ(neuron::summarize_tensors(neuron::Gguf(g.path)), std::string("no tensors\n"));
}
NEURON_TEST(options_limit_output) {
  fixture::File f(sample());
  const neuron::Gguf file(f.path);
  neuron::InspectOptions options;
  options.tensors = false; options.array_items = 2; options.string_bytes = 3;
  const auto text = neuron::describe_gguf(file, options);
  CHECK(!has(text, "tensors:"));
  CHECK(has(text, "[i32 x 10] 0, -1, ... 8 more\n"));
  CHECK(has(text, "\"say\"... (10 bytes)"));
  // A cut never splits a multibyte character.
  fixture::Bytes b; b.header(0, 1); b.text("general.name", "ab\xe2\x98\x83");
  fixture::File g(b);
  options.string_bytes = 4;
  CHECK(has(neuron::describe_gguf(neuron::Gguf(g.path), options), "\"ab\"... (5 bytes)"));
  CHECK_EQ(neuron::tensor_type_name(12), std::string("Q4_K"));
  CHECK_EQ(neuron::tensor_type_name(99), std::string("type 99"));
  CHECK_EQ(neuron::gguf_type_name(neuron::GgufType::u64), std::string_view("u64"));
}
