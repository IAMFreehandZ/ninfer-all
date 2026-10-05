#include "ops/device_profile_fixture.h"
#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer::test::linear_add;
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        ninfer::test::install_device_profile_fixture_from_env();
        constexpr std::array<std::int32_t, 6> boundaries{9, 17, 33, 65, 129, 193};
        constexpr std::array<std::int32_t, 26> widths{
            1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 24, 32, 40, 63, 64, 95, 100, 127, 128,
            192, 256, 300, 385, 600, 1007, 1024};
        constexpr std::array<std::int32_t, 7> graphs{1, 4, 17, 33, 129, 193, 1024};
        constexpr std::array<std::int32_t, 3> full_widths{1, 4, 8};
        int failures = 0;
        for (const std::int32_t input_rows : {6144, 17408}) {
            failures += run_shape("T2_A8 LinearAdd", WeightFormat::T2G128F16S,
                                  {5120, input_rows, static_cast<std::uint32_t>(1901 + input_rows),
                                   boundaries, widths, graphs});
            failures += run_shape("T2_A8 LinearAdd full output", WeightFormat::T2G128F16S,
                                  {5120, input_rows, static_cast<std::uint32_t>(1911 + input_rows),
                                   {}, full_widths, {}, true});
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A8 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A8 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
