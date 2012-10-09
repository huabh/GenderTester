#include "globalHeader.hpp"
#include "faceProcessor.hpp"
#include "images2Columns.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
int main(int argc, char **argv) {
IplImage *inputImg;
trainImage sample;  
cvNamedWindow("With COM", CV_WINDOW_AUTOSIZE);
// CvCapture* capture = 0;
// capture = cvCreateCameraCapture(-1);
// if(!capture){
// return -1;
// }
// inputImg = cvQueryFrame(capture);
#include <opencv2/ml/ml.hpp>
float result;
initializeFaceProcessor();
CvMat* SampleMatrix;
CvMat* PjMatrix=(CvMat*)cvLoad("/home/umut/projects/fastTrainer/build/ProjectionMatrix.xml");
int newDimension=PjMatrix->cols;
// int newDimension;
CvMat* allFeatures;
 CvMat* LDAMatrix=cvCreateMat(newDimension,1,CV_32F);
// CvBoost booster;
// 
// booster.load("/home/umut/projects/fastTrainer/build/Booster.dat");
int trans=CV_GEMM_A_T;

CvSVM SVM;
SVM.load("/home/umut/projects/fastTrainer/build/SVM_CLASS.dat");
// Grab the next frame from the camera.
// while((inputImg = cvQueryFrame(capture))  != NULL ){
for (int i=1;i<argc;i++){
    inputImg=cvLoadImage(argv[i]);


  
      if(processFace(inputImg,  sample.FaceImage, sample.MouthImage, sample.NoseImage, sample.EyeImage, 0))  
      {
	 sample.LBPHF=LBP_HF(sample.FaceImage,sample.nonUniform,sample.complete); //Pass through the LBPHF
// 	  sample.EyeImage=filterGabor(sample.EyeImage);
// 	  sample.NoseImage=filterGabor(sample.NoseImage);
// 	  sample.MouthImage=filterGabor(sample.MouthImage);
	  mat2Col(&sample,0,1,0,SampleMatrix);
	//  newDimension=SampleMatrix->rows;
	  allFeatures=cvCreateMat(1,35+2+newDimension,CV_32F);
	  cvGEMM(PjMatrix,SampleMatrix,1,NULL,0,LDAMatrix,trans);
	   cvSetReal1D(allFeatures,0,sample.complete);
	  cvSetReal1D(allFeatures,1,sample.nonUniform);      
	  for (int j=0;j<35;j++)
	    cvSetReal1D(allFeatures,2+j,sample.LBPHF[j]);
	  for (int j=0;j
	    <newDimension;j++)
	//    cvSetReal1D(allFeatures,37+j,cvGetReal1D(SampleMatrix,j));
	  cvSetReal1D(allFeatures,37+j,cvGetReal1D(LDAMatrix,j));
	//  cout<< "feature Size: "<< allFeatures->cols << "\n";
	//  result=booster.predict(allFeatures,0,booster.get_weak_response());
	  result=SVM.predict(allFeatures);
	  if (result==0)
	  {
	    cvRectangle(sample.FaceImage,cvPoint(2,2),cvPoint(sample.FaceImage->width-2,sample.FaceImage->height-2),cvScalar(255,0,0),3);
	    printf("Result is male\n");
	  }
	  else
	  {
	    cvRectangle(sample.FaceImage,cvPoint(2,2),cvPoint(sample.FaceImage->width-2,sample.FaceImage->height-2),cvScalar(0,0,255),3);
	    printf("Result is female\n");
	  }
	
      cvShowImage("With COM",sample.FaceImage);
     char c=cvWaitKey(0);
//      char c=cvWaitKey(5);
//       if (c==27) break;
 }
}
if (strcmp(argv[1],"1"))
cvReleaseImage( &inputImg);
}
