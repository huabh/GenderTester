PROJECT(gendertester)
include_directories(/usr/local/include/opencv)
include_directories(/usr/local/include/eigen3/Eigen)
link_libraries(/usr/local/lib/libopencv_highgui.so)
link_libraries(/usr/local/lib/libopencv_core.so)
link_libraries(/usr/local/lib/libopencv_features2d.so)
link_libraries(/usr/local/lib/libopencv_imgproc.so)
link_libraries(/usr/local/lib/libopencv_video.so)
link_libraries(/usr/local/lib/libopencv_ts.so)
link_libraries(/usr/local/lib/libopencv_objdetect.so)
link_libraries(/usr/local/lib/libopencv_calib3d.so)
include_directories(/usr/include/libface)
link_libraries(/usr/lib64/libface.so)
ADD_EXECUTABLE(gendertester CvGabor.cpp faceProcessor.cpp images2Columns.cpp LBP-HF.cpp main.cpp)

