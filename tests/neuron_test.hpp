// A tiny test framework: test registration, CHECK/REQUIRE macros, and a main
// that runs every test (or the ones named on the command line).
//
//   NEURON_TEST(shape_numel) { CHECK_EQ(Shape({2, 3}).numel(), 6); }
//
// Define NEURON_TEST_MAIN in exactly one translation unit per test binary.
#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace neuron::test {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// Thrown by REQUIRE to stop the current test.
struct AbortTest {};

inline int& failures_in_case() {
  static int n = 0;
  return n;
}

inline void report(const char* file, int line, const std::string& msg) {
  std::fprintf(stderr, "  %s:%d: %s\n", file, line, msg.c_str());
  ++failures_in_case();
}

template <typename A, typename B>
std::string describe(const char* op, const char* ea, const char* eb, const A& a, const B& b) {
  std::ostringstream os;
  os << "expected " << ea << ' ' << op << ' ' << eb << ", got " << a << " vs " << b;
  return os.str();
}

inline int run(int argc, char** argv) {
  int failed = 0;
  int ran = 0;
  for (const Case& c : registry()) {
    bool selected = argc <= 1;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], c.name) == 0) {
        selected = true;
      }
    }
    if (!selected) {
      continue;
    }
    ++ran;
    failures_in_case() = 0;
    try {
      c.fn();
    } catch (const AbortTest&) {
    } catch (const std::exception& e) {
      report(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      report(__FILE__, __LINE__, "unexpected non-standard exception");
    }
    if (failures_in_case() != 0) {
      ++failed;
      std::fprintf(stderr, "[FAIL] %s\n", c.name);
    } else {
      std::printf("[ ok ] %s\n", c.name);
    }
  }
  if (ran == 0) {
    std::fprintf(stderr, "no tests matched\n");
    return 1;
  }
  std::printf("%d/%d passed\n", ran - failed, ran);
  return failed == 0 ? 0 : 1;
}

}  // namespace neuron::test

#define NEURON_TEST_CAT2(a, b) a##b
#define NEURON_TEST_CAT(a, b) NEURON_TEST_CAT2(a, b)

#define NEURON_TEST(name)                                                        \
  static void NEURON_TEST_CAT(neuron_test_, name)();                             \
  static const ::neuron::test::Registrar NEURON_TEST_CAT(neuron_reg_, name){     \
      #name, &NEURON_TEST_CAT(neuron_test_, name)};                              \
  static void NEURON_TEST_CAT(neuron_test_, name)()

#define NEURON_CHECK_IMPL(cond, on_fail)                                         \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ::neuron::test::report(__FILE__, __LINE__, "check failed: " #cond);        \
      on_fail;                                                                   \
    }                                                                            \
  } while (0)

#define NEURON_CMP_IMPL(a, b, op, on_fail)                                       \
  do {                                                                           \
    const auto& neuron_a_ = (a);                                                 \
    const auto& neuron_b_ = (b);                                                 \
    if (!(neuron_a_ op neuron_b_)) {                                             \
      ::neuron::test::report(                                                    \
          __FILE__, __LINE__,                                                    \
          ::neuron::test::describe(#op, #a, #b, neuron_a_, neuron_b_));          \
      on_fail;                                                                   \
    }                                                                            \
  } while (0)

#define CHECK(cond) NEURON_CHECK_IMPL(cond, (void)0)
#define REQUIRE(cond) NEURON_CHECK_IMPL(cond, throw ::neuron::test::AbortTest{})
#define CHECK_EQ(a, b) NEURON_CMP_IMPL(a, b, ==, (void)0)
#define CHECK_NE(a, b) NEURON_CMP_IMPL(a, b, !=, (void)0)
#define CHECK_LT(a, b) NEURON_CMP_IMPL(a, b, <, (void)0)
#define CHECK_LE(a, b) NEURON_CMP_IMPL(a, b, <=, (void)0)
#define REQUIRE_EQ(a, b) NEURON_CMP_IMPL(a, b, ==, throw ::neuron::test::AbortTest{})

// Passes when `expr` throws an exception of type `exc`.
#define CHECK_THROWS_AS(expr, exc)                                               \
  do {                                                                           \
    bool neuron_threw_ = false;                                                  \
    try {                                                                        \
      (void)(expr);                                                              \
    } catch (const exc&) {                                                       \
      neuron_threw_ = true;                                                      \
    } catch (...) {                                                              \
    }                                                                            \
    if (!neuron_threw_) {                                                        \
      ::neuron::test::report(__FILE__, __LINE__,                                 \
                             "expected " #expr " to throw " #exc);               \
    }                                                                            \
  } while (0)

#ifdef NEURON_TEST_MAIN
int main(int argc, char** argv) { return ::neuron::test::run(argc, argv); }
#endif
