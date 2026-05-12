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

add_executable(test_tracking test_tracking.cpp)
target_link_libraries(test_tracking PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_tracking COMMAND test_tracking)

add_executable(test_ba test_ba.cpp)
target_link_libraries(test_ba PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_ba COMMAND test_ba)

add_executable(test_map test_map.cpp)
target_link_libraries(test_map PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_map COMMAND test_map)

add_executable(test_local_ba test_local_ba.cpp)
target_link_libraries(test_local_ba PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_local_ba COMMAND test_local_ba)

add_executable(test_orb_vocabulary test_orb_vocabulary.cpp)
target_link_libraries(test_orb_vocabulary PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_orb_vocabulary COMMAND test_orb_vocabulary
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_keyframe_database test_keyframe_database.cpp)
target_link_libraries(test_keyframe_database PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_keyframe_database COMMAND test_keyframe_database
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_sim3_solver test_sim3_solver.cpp)
target_link_libraries(test_sim3_solver PRIVATE sslam GTest::gtest GTest::gtest_main)
add_test(NAME test_sim3_solver COMMAND test_sim3_solver)
