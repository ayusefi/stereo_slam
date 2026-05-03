find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, skipping tests")
    return()
endif()

add_executable(test_kitti_loader test_kitti_loader.cpp)
target_link_libraries(test_kitti_loader PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_kitti_loader COMMAND test_kitti_loader)

add_executable(test_orb_extractor test_orb_extractor.cpp)
target_link_libraries(test_orb_extractor PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_orb_extractor COMMAND test_orb_extractor)

add_executable(test_feature_matcher test_feature_matcher.cpp)
target_link_libraries(test_feature_matcher PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_feature_matcher COMMAND test_feature_matcher)
