#define NEURON_TEST_MAIN
#include "neuron/gguf.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <cmath>
#include <limits>

using fixture::Bytes;
using fixture::File;
using neuron::Gguf;

namespace {
Bytes valid(std::uint32_t version = 3) {
  Bytes b;
  b.header(1, 1, version);
  b.uint("general.alignment", 64);
  b.tensor("test", {2, 2}, 0, 0);
  b.align(64);
  for (float v : {1.0f, 2.0f, -3.0f, 4.0f}) b.put(v);
  return b;
}
void rejects(const Bytes& bytes, const std::string& reason = "GGUF:") {
  File f(bytes);
  try { Gguf g(f.path); CHECK(false); }
  catch (const std::runtime_error& e) { CHECK(std::string(e.what()).find(reason) != std::string::npos); }
}
}
NEURON_TEST(reads_aligned_f32_tensor_and_v2_v3) {
  for (const auto version : {2u, 3u}) {
    File file(valid(version)); Gguf g(file.path);
    CHECK_EQ(g.alignment(), 64u);
    CHECK_EQ(g.tensors().size(), 1u);
    CHECK_EQ(g.tensor("test").dimensions[0], 2u);
    const auto values = g.tensor_f32("test");
    CHECK_EQ(values[0], 1.0f); CHECK_EQ(values[2], -3.0f);
    CHECK_EQ(g.tensor_data(g.tensor("test")).size(), 16u);
    CHECK(g.find("missing") == nullptr);
    CHECK_THROWS_AS(g.at("missing"), std::runtime_error);
    CHECK_THROWS_AS(g.tensor("missing"), std::runtime_error);
  }
}
NEURON_TEST(all_metadata_types_and_arrays) {
  Bytes b; b.header(0, 16);
  b.key("u8", 0); b.put<std::uint8_t>(255);
  b.key("i8", 1); b.put<std::int8_t>(-8);
  b.key("u16", 2); b.put<std::uint16_t>(65535);
  b.key("i16", 3); b.put<std::int16_t>(-16);
  b.uint("u32", 123);
  b.key("i32", 5); b.put<std::int32_t>(-32);
  b.key("f32", 6); b.put(1.25f);
  b.key("bool", 7); b.put<std::uint8_t>(1);
  b.text("string", std::string("a\0b", 3));
  b.key("u64", 10); b.put<std::uint64_t>(1ull << 63);
  b.key("i64", 11); b.put<std::int64_t>(-64);
  b.key("f64", 12); b.put(1.125);
  b.key("strings", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(2); b.string("one"); b.string("two");
  b.key("ints", 9); b.put<std::uint32_t>(5); b.put<std::uint64_t>(2); b.put<std::int32_t>(-1); b.put<std::int32_t>(2);
  b.key("floats", 9); b.put<std::uint32_t>(6); b.put<std::uint64_t>(1); b.put(0.5f);
  b.key("empty", 9); b.put<std::uint32_t>(0); b.put<std::uint64_t>(0);
  File f(b); Gguf g(f.path);
  CHECK_EQ(g.at("u8").uint(), 255u); CHECK_EQ(g.at("u16").uint(), 65535u);
  CHECK_EQ(g.at("u32").uint(), 123u); CHECK_EQ(g.at("u64").uint(), 1ull << 63);
  for (const auto* k : {"i8", "i16", "i32", "i64"}) CHECK(std::get<std::int64_t>(g.at(k).data) < 0);
  CHECK_EQ(g.at("f32").number(), 1.25); CHECK_EQ(g.at("f64").number(), 1.125);
  CHECK(g.at("bool").boolean()); CHECK_EQ(g.at("string").string().size(), 3u);
  CHECK_EQ(g.at("strings").array()[1].string(), std::string("two"));
  CHECK_EQ(std::get<std::int64_t>(g.at("ints").array()[0].data), -1);
  CHECK_EQ(g.at("floats").array()[0].number(), 0.5); CHECK(g.at("empty").array().empty());
  CHECK_THROWS_AS(g.at("string").uint(), std::runtime_error);
}
NEURON_TEST(f16_upcast_including_subnormals) {
  Bytes b; b.header(1, 0); b.tensor("half", {8}, 1, 0); b.align();
  for (const auto v : {0x0000, 0x8000, 0x3c00, 0xc000, 0x0001, 0x7bff, 0x7c00, 0x7e00}) b.put(static_cast<std::uint16_t>(v));
  File f(b); Gguf g(f.path); const auto a = g.tensor_f32("half");
  CHECK_EQ(a[0], 0.0f); CHECK(std::signbit(a[1])); CHECK_EQ(a[2], 1.0f); CHECK_EQ(a[3], -2.0f);
  CHECK_EQ(a[4], std::ldexp(1.0f, -24)); CHECK_EQ(a[5], 65504.0f);
  CHECK(std::isinf(a[6])); CHECK(std::isnan(a[7]));
}
NEURON_TEST(every_truncation_is_rejected) {
  const auto good = valid();
  for (std::size_t n = 0; n < good.data.size(); ++n) {
    auto b = good; b.data.resize(n); rejects(b);
  }
}
NEURON_TEST(bad_magic_version_counts_and_strings) {
  auto b = valid(); b.patch<std::uint32_t>(0, 0); rejects(b, "magic");
  for (auto version : {0u, 1u, 4u}) { b = valid(); b.patch(4, version); rejects(b, "version"); }
  for (auto offset : {8u, 16u}) { b = valid(); b.patch<std::uint64_t>(offset, UINT64_MAX); rejects(b, "count"); }
  b = valid(); b.patch<std::uint64_t>(24, UINT64_MAX); rejects(b, "string");
  Bytes a; a.header(0, 1); a.key("array", 9); a.put<std::uint32_t>(8); a.put<std::uint64_t>(UINT64_MAX);
  rejects(a, "array");
}
NEURON_TEST(bad_alignment_offsets_dimensions_and_types) {
  for (auto alignment : {0u, 3u}) {
    Bytes b; b.header(0, 1); b.uint("general.alignment", alignment); rejects(b, "alignment");
  }
  for (auto offset : {std::uint64_t{1}, std::uint64_t{4096}, UINT64_MAX}) {
    Bytes b; b.header(1, 0); b.tensor("x", {1}, 0, offset); b.align(); b.put(0.0f); rejects(b);
  }
  for (auto dim : {std::uint64_t{0}, UINT64_MAX}) {
    Bytes b; b.header(1, 0); b.tensor("x", {dim, 2}, 0, 0); b.align(); b.put(0.0f); rejects(b);
  }
  Bytes b; b.header(1, 0); b.tensor("x", {1}, 99, 0); b.align(); b.put(0.0f); rejects(b, "type");
  Bytes a; a.header(0, 1); a.key("x", 99); a.put<std::uint32_t>(0); rejects(a, "type");
  Bytes c; c.header(0, 1); c.key("x", 7); c.put<std::uint8_t>(2); rejects(c, "boolean");
}
NEURON_TEST(duplicates_and_overlap_are_rejected) {
  Bytes b; b.header(0, 2); b.uint("a", 1); b.uint("a", 2); rejects(b, "duplicate");
  for (const auto* name : {"x", "y"}) {
    Bytes a; a.header(2, 0); a.tensor("x", {1}, 0, 0); a.tensor(name, {1}, 0, 0);
    a.align(); a.put(1.0f); rejects(a);
  }
}
