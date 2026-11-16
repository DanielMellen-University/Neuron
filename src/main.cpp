#include <cstdio>
#include <cstring>

#include "neuron/version.hpp"

namespace {

void print_usage() {
  std::printf(
      "usage: neuron [--version] [--help]\n"
      "\n"
      "Neuron is in early development. Model loading and generation are not\n"
      "implemented yet; see ROADMAP.md.\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("neuron %s (%s)\n", neuron::version_string().data(),
                neuron::compiled_isa().data());
    return 0;
  }
  if (argc == 1 || (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                                  std::strcmp(argv[1], "-h") == 0))) {
    print_usage();
    return 0;
  }
  std::fprintf(stderr, "neuron: unknown arguments\n");
  print_usage();
  return 2;
}
