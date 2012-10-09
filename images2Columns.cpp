#include "images2Columns.hpp"

vector<int> mat2Col(trainImage* Men,trainImage* Women,int MCount,int WCount,CvMat* &Result)
{
  IplImage* tempE=Men[0].EyeImage;
  IplImage* tempM=Men[0].MouthImage;
  IplImage* tempN=Men[0].NoseImage;
  CvSize szE=cvGetSize(tempE);
  CvSize szM=cvGetSize(tempM);
  CvSize szN=cvGetSize(tempN);
  
  vector<int> labels;
   Result=cvCreateMat(szE.height*szE.width+szM.height*szM.width+szN.height*szN.width,MCount+WCount, CV_32F);
   
   for (int i=0;i<MCount;i++)
   {
    tempE=Men[i].EyeImage; 
    tempM=Men[i].MouthImage;
    tempN=Men[i].NoseImage;
      for (int x=0;x<szE.width;x++)
      {
	for (int y=0;y<szE.height;y++)
	{
	  //  cout<<"Row of Big Matrix:" << i << "  Row and Column of Small: "<<  x <<"  " <<y<< "\n";
	      cvSetReal2D(Result,x*szE.height+y,i,cvGetReal2D(tempE,y,x));
	      
	}
      } 
      for (int x=0;x<szM.width;x++)
      {
	for (int y=0;y<szM.height;y++)	
	{
	  //  cout<<"Row of Big Matrix:" << i << "  Row and Column of Small: "<<  x <<"  " <<y<< "\n";
	      cvSetReal2D(Result,szE.height*szE.width+x*szM.height+y,i,cvGetReal2D(tempM,y,x));
	      
	}
      }  
      
      for (int x=0;x<szN.width;x++)
      {
	for (int y=0;y<szN.height;y++)
	{
	  //  cout<<"Row of Big Matrix:" << i << "  Row and Column of Small: "<<  x <<"  " <<y<< "\n";
	      cvSetReal2D(Result,szE.height*szE.width+szM.height*szM.width+x*szN.height+y,i,cvGetReal2D(tempN,y,x));
	      
	}
      }  
  //    printf("complete:%d\n",i    );
      labels.push_back(0);
    }
//  cout<< "COlumn Count:"<<Result->cols <<"\n" ;

   
  for (int i=0;i<WCount;i++)
   {
    tempE=Women[i].EyeImage; 
    tempM=Women[i].MouthImage;
    tempN=Women[i].NoseImage;
      for (int x=0;x<szE.width;x++)
      {
	for (int y=0;y<szE.height;y++)
	{
	      cvSetReal2D(Result,x*szE.height+y,i+MCount,cvGetReal2D(tempE,y,x));
	     
	}
      } 
      for (int x=0;x<szM.width;x++)
      {
	for (int y=0;y<szM.height;y++)
	{
	      cvSetReal2D(Result,szE.height*szE.width+x*szM.height+y,i+MCount,cvGetReal2D(tempM,y,x));
	     
	}
      } 
      for (int x=0;x<szN.width;x++)
      {
	for (int y=0;y<szN.height;y++)
	{
	      cvSetReal2D(Result,szE.height*szE.width+szM.height*szM.width+x*szN.height+y,i+MCount,cvGetReal2D(tempN,y,x));
	     
	}
      } 
       labels.push_back(1);
    }
  
  return labels;
}