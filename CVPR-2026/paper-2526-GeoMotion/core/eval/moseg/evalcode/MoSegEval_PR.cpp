#include <algorithm>
#include <string.h>
#include <CVector.h>
#include <CMatrix.h>
#include <CTensor.h>
#include"hungarian.h"

//#define DEBUG

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

const int mR[100] =    {255,200, 10,  0,255,  0,  0,180,255,180,  0,170,100,240,  0,255, 10,200, 90, 10,  0,180, 70,100,255, 30,100,  0,100,255, 50,120,140, 70, 70,100,235,180,200,210,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59};
const int mG[100] =    { 10,190, 10,180,  0,255,255,  0,130,  0,  0,170,100,100,100, 70, 10,200,  0,100,150, 75, 70, 30,100,  0,  0,  0,255,100, 80,180,100,100,140,200,235,200, 70,160,
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60};
const int mB[100] =    { 10,  0,255,  0,255,255,  0,  0,  0,180,130,170,100, 80,255, 30, 10, 70, 90, 50, 25, 10,255,170,170,200,  0, 75,100,100,200,120,  0,  0,140,200,100,  0,250,210,
2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61};
const int mGray[40] = {  0,255,155, 40,200,100,175,220, 10, 20, 70, 80, 90,110,120,210,130,  5, 85,190,195,220,230,240,105,135,155, 75,175, 35,128,202,166,133,111, 99, 54,123,244, 21};


/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */




using namespace std;
class CPoint {
public:
  CPoint() {}
  int x,y,frame;
};

class CSimpleTrack {
public:
  CSimpleTrack() {}
  int mLabel;
  CVector<CPoint> mPoints;
};

class CCoverage {
public:
  CCoverage() {}
  CCoverage(int aRegion, int aCoverage) : mRegion(aRegion),mCoverage(aCoverage) {}
  int mRegion;
  int mCoverage;
};

bool operator<(const CCoverage& a, const CCoverage& b) {
  return a.mCoverage < b.mCoverage;
}

int mTotalFrameNo;
int mLabeledFramesNo;
int mRegionNo;
int mClusterNo;

CVector<int> mLabeledFrames;
CVector<CMatrix<float> > mRegions;      // ground truth regions
CVector<CMatrix<float> > mRegionsProb;  // ground truth probabilities
int mEvaluateFrames;
CVector<CSimpleTrack> mTracks;
CVector<int> mColor2Region;

bool readTracks(char* aFilename) {
  std::ifstream aFile(aFilename);
  if (!aFile.is_open()) {
    std::cerr << aFilename << " was not found." << std::endl;
    return false;
  }
  int aLength;
  aFile >> aLength;
  if (aLength == 10) mEvaluateFrames = 10;
  else if (aLength == 50) mEvaluateFrames = 50;
  else if (aLength == 200) mEvaluateFrames = 200;
  else if (aLength == mTotalFrameNo) mEvaluateFrames = mTotalFrameNo; 
  else {
    std::cerr << "Length of tracked sequence does not match a typical evaluation length (10,50,200,all frames)." << std::endl;
    return false;
  }
  int aTrackNo;
  aFile >> aTrackNo;
  mTracks.setSize(aTrackNo);
  for (int i = 0; i < aTrackNo; i++) {
    aFile >> mTracks(i).mLabel;
    int aSize;
    aFile >> aSize;
    mTracks(i).mPoints.setSize(aSize);
    float x,y,frame;
    for (int j = 0; j < aSize; j++) {
      aFile >> x >> y >> frame;
      mTracks(i).mPoints(j).x = (int)(x+0.5f);
      mTracks(i).mPoints(j).y = (int)(y+0.5f);
      mTracks(i).mPoints(j).frame = (int)frame;
    }
  }
  // Count number of clusters
  mClusterNo = 0;
  for (int i = 0; i < mTracks.size(); i++)
    if (mTracks(i).mLabel+1 > mClusterNo) mClusterNo = mTracks(i).mLabel+1;
  return true;
}

int** convertFormat(CMatrix<int> &mat)
{
  int** r;
  int i,j;
  int rows,cols;
  rows = mat.ySize();
  cols = mat.xSize();
  r = new int*[rows*sizeof(int*)];
  for(i=0;i<rows;i++)
  {
    r[i] = new int[cols*sizeof(int)];
  }
  for(i=0;i<rows;i++)
    for(j=0;j<cols;j++){
      r[i][j]=-mat(j,i);
    }  

  return r;

}

void computeFmeasure(CMatrix<float> precision,CMatrix<float> recall, CMatrix<int>& FmeasureMat,string path)
{
  int matSize = FmeasureMat.xSize(); // xsize = ysize
  CMatrix<float> Ffl(matSize,matSize,0);
  Ffl.fill(0);
  for(int y=0;y<precision.ySize();y++)
    for(int x=0;x<precision.xSize();x++){
      if(precision(x,y)!=0 && recall(x,y)!=0){
        float F = (2*precision(x,y)*recall(x,y))/(precision(x,y)+recall(x,y)); // remove variable F later
        Ffl(x,y)=F;
        F = F*10000;
        int Fint = (int)F;
        FmeasureMat(x,y) = Fint;
      }
    }
}


int main(int argc, char* args[]) {
  if (argc <= 2) {
    std::cout << "Usage: MoSegEval shotDef.dat yourtracks.dat" << std::endl;
    return -1;
  }
  // Read definition file and ground truth -------------------------------------
  std::string s = args[1];
  std::string aPath = s;
  if (aPath.find_last_of('/') < aPath.length()) aPath.erase(aPath.find_last_of('/'),aPath.length());
  else aPath = ".";
  aPath += '/';
  std::ifstream aPropFile(s.c_str());
  if (!aPropFile.is_open()) {
    std::cerr << "Definition file " << s.c_str() << "  not found." << std::endl;
    return -1;
  }
  // Read header
  char dummy[300];
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  // Number of regions
  aPropFile.getline(dummy,300);
  aPropFile >> mRegionNo; aPropFile.getline(dummy,300);
  mColor2Region.setSize(16777216);
  CMatrix<float> aPenalty(mRegionNo,mRegionNo);
  // Region color
  for (int i = 0; i < mRegionNo; i++) {
    aPropFile.getline(dummy,300);
    int a;
    aPropFile >> a; aPropFile.getline(dummy,300);
    mColor2Region(a) = i;
  }
  // Confusion penalty matrix
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  for (int j = 0; j < mRegionNo; j++)
    for (int i = 0; i < mRegionNo; i++)
      aPropFile >> aPenalty(i,j);
  // Number of frames in shot
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  aPropFile >> mTotalFrameNo; aPropFile.getline(dummy,300);
  // Number of labeled frames
  aPropFile.getline(dummy,300);
  aPropFile >> mLabeledFramesNo; aPropFile.getline(dummy,300);
  mLabeledFrames.setSize(mLabeledFramesNo);
  mRegions.setSize(mLabeledFramesNo);
  mRegionsProb.setSize(mLabeledFramesNo);
  // Read frame number and annotation
  for (int i = 0; i < mLabeledFramesNo; i++) {
    aPropFile.getline(dummy,300);
    aPropFile >> mLabeledFrames(i); aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    std::string s;
    aPropFile >> s;
    std::string ending = s;
    ending.erase(0,ending.find_last_of('.')+1);
    // if s is pgm file (Berkeley Motion Segmentation Benchmark)
    if (strcmp(ending.c_str(),"pgm") == 0) {
      mRegions(i).readFromPGM((aPath+s).c_str());
      mRegionsProb(i).setSize(mRegions(i).xSize(),mRegions(i).ySize());
      mRegionsProb(i).fill(1.0);
    } else {
    // else if s is ppm (Freiburg Motion Segmentation Benchmark)
      if (strcmp(ending.c_str(),"ppm") == 0) {
        CTensor<float> aCRegion;
        aCRegion.readFromPPM((aPath+s).c_str());
        mRegions(i).setSize(aCRegion.xSize(), aCRegion.ySize());
        for (int y=0; y<aCRegion.ySize(); y++) {
          for (int x=0; x<aCRegion.xSize(); x++) {
            // scale value is red*256^2 + green*256^1 + blue*256^0
            mRegions(i)(x,y) = aCRegion(x,y,0)*65536 + aCRegion(x,y,1)*256 + aCRegion(x,y,2);
          }
        }
        // load probability map
        s.erase(s.find_last_of('.'),s.length());
        s += ".pgm";
        mRegionsProb(i).readFromPGM((aPath+s).c_str());
        for (int j=0; j<mRegionsProb(i).size(); j++) {
          mRegionsProb(i).data()[j] /= 255.0;
        }
      } else { // none of both cases -> error
        std::cerr << "Error while reading ground truth file: " << " wrong ending" << std::endl;
      }
    }
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
  }
  // Read tracks ---------------------------------------------------------------
  if (!readTracks(args[2])) return -1;
  // Evaluate ------------------------------------------------------------------
  CMatrix<float> aRegionClusterOverlap(mRegionNo,mClusterNo,0); // overlap of cluster i and region j
  CVector<float> aClusterSize(mClusterNo,0); // size of each cluster
  CVector<float> aRegionSize(mRegionNo,0); // occupied pixel of each region by the clustering
  CVector<float> aRegionCounter(mRegionNo,0); // total size of each region
  int aUsedLabeledFrames = 0;
  int aTotalCoverage = 0;
  // Measure coverage of regions (ground truth) by clusters (estimated track labels)
  for (int t = 0; t < mLabeledFramesNo; t++) {
    if (mLabeledFrames(t) > mEvaluateFrames && mEvaluateFrames > 0) break;
    aUsedLabeledFrames++;
    for (int i = 0; i < mRegions(t).size(); i++) {
      aRegionCounter(mColor2Region((int)(mRegions(t).data()[i]))) += mRegionsProb(t).data()[i];
    }
    CMatrix<bool> aOccupied(mRegions(t).xSize(),mRegions(t).ySize(),false);
    for (int i = 0; i < mTracks.size(); i++) {
      if (mTracks(i).mPoints(0).frame > mLabeledFrames(t)
        || mTracks(i).mPoints(mTracks(i).mPoints.size()-1).frame < mLabeledFrames(t)) continue;
      int t2 = mLabeledFrames(t)-mTracks(i).mPoints(0).frame;
      int x = mTracks(i).mPoints(t2).x;
      int y = mTracks(i).mPoints(t2).y;
      if (x < 0 || y < 0 || x >= mRegions(t).xSize() || y >= mRegions(t).ySize()) continue;
      int aRegion = mColor2Region((int)mRegions(t)(x,y));
      aRegionClusterOverlap(aRegion,mTracks(i).mLabel) += mRegionsProb(t)(x,y);
      aClusterSize(mTracks(i).mLabel) += mRegionsProb(t)(x,y);
      aRegionSize(mColor2Region((int)(mRegions(t)(x,y)))) += mRegionsProb(t)(x,y);
      // Count double occupation of pixels, so it does not increase the density
      if (aOccupied(x,y) == false) aTotalCoverage++;
      aOccupied(x,y) = true;
    }
  }
  #ifdef DEBUG
  std::cout << "number of regions: " << mRegionNo << std::endl;
  std::cout << "number of clusters: " << mClusterNo << std::endl;
  std::cout << std::endl;
  std::cout << "Region cluster overlap: " << std::endl; 
  for (int j=0; j<aRegionClusterOverlap.ySize(); ++j) {
    std::cout << "[" << std::flush;
    for (int i=0; i<aRegionClusterOverlap.xSize(); ++i)
    {
      printf("%12.2f ", aRegionClusterOverlap(i,j));
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Cluster size: " << std::endl;
  for (int i=0; i<aClusterSize.size(); ++i)
  {
    std::cout << "cluster size(" << i << ") = " << aClusterSize(i) << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Region size (occupied pixels by clustering): " << std::endl;
  for (int i=0; i<aRegionSize.size(); ++i)
  {
    std::cout << "region size(" << i << ") = " << aRegionSize(i) << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Total Region size: " << std::endl;
  for (int i=0; i<aRegionSize.size(); ++i)
  {
    std::cout << "total region size(" << i << ") = " << aRegionCounter(i) << std::endl;
  }
  std::cout << std::endl;
  #endif

  // Compute final numbers
  CVector<float> densityRegion(mRegionNo,0.0f);
  float avgDensityRegion = 0.0f;
  std::vector<int> visibleRegionToRegion;
  for(int iter=0; iter<mRegionNo; iter++) {
    if (aRegionCounter(iter) < 1e-4f) continue;
    visibleRegionToRegion.push_back(iter);
    avgDensityRegion += aRegionSize(iter)/aRegionCounter(iter);
  }
  int aAllVisibleRegions = visibleRegionToRegion.size();
  avgDensityRegion = avgDensityRegion/(float) aAllVisibleRegions;

  #ifdef DEBUG
  std::cout << "Assignment visible region to region" << std::endl;
  std::cout << "There are " << aAllVisibleRegions << " visible regions" << std::endl;
  for(int i=0; i<aAllVisibleRegions; i++) 
  {
    std::cout << "visible region(" << i << ") -> region(" << visibleRegionToRegion[i] << ")" << std::endl;
  }
  std::cout << std::endl;
  #endif

  CVector<float> aPrecision(mClusterNo, 0.0f);
  CVector<float> aRecall(aAllVisibleRegions, 0.0f);
  CVector<bool> aObject(aAllVisibleRegions, false);
  CMatrix<float> P(aAllVisibleRegions,mClusterNo,0);
  CMatrix<float> R(aAllVisibleRegions,mClusterNo,0);

  int aObjectCount = -1; // subtract background
  float aObjectThreshold = 0.9f;
  if (argc >= 4)
    aObjectThreshold = atof(args[3]); // 0.9f;
  int aVisibleRegions = aAllVisibleRegions-1; // subtract background
  for (unsigned int j2=0; j2<aAllVisibleRegions; ++j2) {
    unsigned int j=visibleRegionToRegion[j2];
    for (int i=0; i<mClusterNo; ++i) {
      float overlap = aRegionClusterOverlap(j,i);
      float regionSize = aRegionSize(j);
      float clusterSize = aClusterSize(i);
      if (clusterSize == 0) continue;
      float precision = overlap / clusterSize;
      P(j2,i) = precision;
      if (regionSize == 0) continue;
      float recall = overlap / regionSize;
      R(j2,i) = recall;
      aPrecision(i) = NMath::max(aPrecision(i), precision);
      aRecall(j2) = NMath::max(aRecall(j2), recall);
    }
  }
  string Ppath = aPath + "P.txt";
  string Rpath = aPath + "R.txt";
  float aDensity = 100.0f*aTotalCoverage/(aUsedLabeledFrames*mRegions(0).size());
  aVisibleRegions = NMath::max(0, aVisibleRegions);

  // compute F-measure for all pairs
  CMatrix<int> aFmeasureMat;
  int costMatSize = (aAllVisibleRegions > mClusterNo)? aAllVisibleRegions : mClusterNo;
  int emptyRegions = (costMatSize == mClusterNo)? 1 : 0;

  aFmeasureMat.setSize(costMatSize,costMatSize);
  aFmeasureMat.fill(0);
  computeFmeasure(P,R,aFmeasureMat,aPath);
  string Fpath = aPath + "Fint.txt";
  //string hungarian = "./hungarian_test " + Fpath;
  //cout << hungarian << endl;
  //system(hungarian.c_str());
  // solve Hungarian method / bipartite graph matching
  hungarian_problem_t p;
  int rows,cols;
  rows = costMatSize; cols = costMatSize;
  int **m = convertFormat(aFmeasureMat);
  int matrix_size = hungarian_init(&p, m , rows,cols, HUNGARIAN_MODE_MINIMIZE_COST) ;
  /* some output */
  fprintf(stderr, "cost-matrix:");
  hungarian_print_costmatrix(&p);

  /* solve the assignement problem */
  hungarian_solve(&p);

  /* some output */
  fprintf(stderr, "assignment:");
  hungarian_print_assignment(&p);
  int **assignment = p.assignment;
  
  // Output results ------------------------------------------------------------
  s = args[2];
  s.erase(s.find_last_of('.'),s.length());
  std::ofstream aOut((s+"Numbers.txt").c_str());
  std::cout << "output file: " << s+"Numbers.txt" << std::endl;
  aOut << "Evaluation results for: " << args[2] << std::endl;
  aOut << "MoSegEval Version 2.0" << std::endl << std::endl;
  aOut << "Number of frames used from the sequence:" << std::endl;
  aOut << mEvaluateFrames << std::endl;
  aOut << "Number of labeled frames in this time window:" << std::endl;
  aOut << aUsedLabeledFrames << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Average region density (in percent):" << std::endl;
  aOut << avgDensityRegion*100 << std::endl;
  aOut << "------------------------------------------------------------------------" << std::endl;
  aOut << "Cluster to Region Assignment with (Precsion, Recall, F-meassure) values:" << std::endl;
  float avgP,avgR,avgF;
  avgP = avgR = avgF = 0;
  int ctr = 0;
  for(int l=0;l<rows;l++) { // use rows instead of mRegionNo to include empty regions
    for(int k=0;k<cols;k++){   // use cols instead of mClusterNo to include empty clusters
      if(assignment[l][k]==1) {
        if (k<aAllVisibleRegions)
          aOut << "Cluster " << l << " => Region " << visibleRegionToRegion[k] << endl;
        else
          aOut << "Cluster " << l << " => Region " << -1 << endl;
        if(emptyRegions && k >= aAllVisibleRegions){
          continue;
        }
        if(!emptyRegions && l >= mClusterNo){
          aOut << 1 << " " << 0 << " " << 0 << " (empty cluster)" << endl;
          avgP += 1;
          ctr++;
          continue;
        }
        
        // F-meassure
        float Fmess = 0.0f;
        if(P(k,l)!=0 && R(k,l)!=0) 
          Fmess = (2*P(k,l)*R(k,l))/(P(k,l)+R(k,l));

        if(P(k,l)!=0 && R(k,l)!=0)
          aOut << P(k,l) << " " << R(k,l) << " " << Fmess <<  endl;
        else
          aOut << 0 << " " << 0 << " " << 0 << endl;
        ctr++;

        // increase number of found objects or not
        if (Fmess > aObjectThreshold)
          aObjectCount ++;

        avgP += P(k,l);
        avgR += R(k,l);
      }
    }
  }
  sprintf(dummy, "%.2f", aObjectThreshold);
  std::string sdummy = dummy;
  avgP = avgP/ctr; avgR = avgR/ctr; 
  avgF = 2.0f*avgP*avgR/(avgP+avgR);
  aObjectCount = NMath::max(0, aObjectCount);
  aOut << "-------------------------------------" << std::endl;
  aOut << "Average Precision, Recall, F-measure:" << std::endl;
  aOut << avgP << " " << avgR << " " << avgF << std::endl;
  aOut << "--------------------------------------------------" << std::endl;
  aOut << "Visible objects in the evaluated part of the shot:" << std::endl;
  aOut << aVisibleRegions << std::endl;
  aOut << "------------------------------------------------------------------------------------" << std::endl;
  aOut << "Extracted objects (#{F-measure > " << sdummy << "}) (excluding background):" << std::endl;
  aOut << aObjectCount << std::endl;
  aOut << "------------------------------" << std::endl;
  aOut << "RGB pixel values for clusters:" << std::endl;
  for(int l=0;l<rows;l++) // use rows instead of mRegionNo to include empty regions
    for(int k=0;k<cols;k++){   // use cols instead of mClusterNo to include empty clusters
      if(assignment[l][k]==1){
        aOut << "Cluster " << l << " => (" << mR[l] << "," << mG[l] << "," << mB[l] << ")" << endl; 
      }
    }

  aOut.close();
  /* free used memory */
  hungarian_free(&p);
  for(int i=0;i<rows;i++)
    delete[] m[i];
  delete[] m;

  return 0;
}
