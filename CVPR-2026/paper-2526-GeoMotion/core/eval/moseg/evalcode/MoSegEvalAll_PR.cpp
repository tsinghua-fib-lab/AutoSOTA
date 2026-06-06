#include <string.h>
#include <string>
#include <fstream>
#include <CVector.h>
#include <CMatrix.h>

int main(int argc, char* args[]) {
  if (argc <= 3) {
    std::cout << "Usage: MoSegEvalAll_PR shotList.txt 10|50|200|all trackList.txt" << std::endl;
    return -1;
  }
  std::string binaryPath = args[0];
  binaryPath.erase(binaryPath.find_last_of('/')+1,binaryPath.length());
  std::ifstream aShotList(args[1]);
  std::ifstream aTrackList(args[3]);
  if (!aShotList.is_open()) {
    std::cerr << args[1] << " could not be opened." << std::endl;
    return -1;
  }
  if (!aTrackList.is_open()) {
    std::cerr << args[3] << " could not be opened." << std::endl;
    return -1;
  }
  int aEvaluationMode;
  if (strcmp(args[2],"all") == 0) aEvaluationMode = -1;
  else aEvaluationMode = atoi(args[2]);
  if (aEvaluationMode != -1 && aEvaluationMode != 10 && aEvaluationMode != 50 && aEvaluationMode != 200) {
    std::cerr << "Evaluation of " << args[2] << " frames is not allowed. Choose 10, 50, 200, or all frames." << std::endl;
    return -1;
  }
  char dummy[300];
  float objectThreshold = 0.9f;
  if (argc >= 5)
    objectThreshold = atof(args[4]);
  sprintf(dummy, "_Fgeq%4.2f", objectThreshold*100.0f);
  // Open output file
  std::string s = args[3];
  s.erase(s.find_last_of('.'),s.length());
  s += dummy; 
  s += "Numbers.txt";
  std::ofstream aOut(s.c_str());
  if (!aOut.is_open()) {
    std::cerr << "Could not write output file." << std::endl;
    return -1;
  }
  float aSumDensity = 0.0f;
  float aSumAvrgDensity = 0.0f;
  float aSumAvrgPrecision = 0.0f;
  float aSumAvrgRecall = 0.0f;
  float aSumAvrgPrecisionSq = 0.0f;
  float aSumAvrgRecallSq = 0.0f;
  int aSumExtracted = 0;
  int aSumVisibles = 0;
  int aSize;
  aShotList >> aSize; aShotList.getline(dummy,300);
  int aCounter = 0;
  for (int shot = 0; shot < aSize; shot++) {
    // Read parts of the definition file ---------------------------------------
    std::string aShotLine;
    aShotList >> aShotLine;
    std::ifstream aPropFile(aShotLine.c_str());
    if (!aPropFile.is_open()) {
      std::cerr << "Definition file " << aShotLine << "  not found." << std::endl;
      return -1;
    }
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    // Number of regions
    aPropFile.getline(dummy,300);
    int aRegionNo;
    aPropFile >> aRegionNo; aPropFile.getline(dummy,300);
    // Number of frames
    for (int i = 0; i < 3*aRegionNo; i++)
      aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    int aTotalFrameNo;
    aPropFile >> aTotalFrameNo; aPropFile.getline(dummy,300);
    // Ignore this shot if it does not have the required number of frames
    if (aEvaluationMode > 0 && aTotalFrameNo < aEvaluationMode) {
      std::cout << "Only " << aTotalFrameNo << " frames. " << aShotLine << " ignored." << std::endl;
      continue;
    }
    // Remove empty lines in list of tracking files ----------------------------
    std::string aTrackLine;
    do {
      aTrackList >> aTrackLine;
      while (aTrackLine[0] == ' ')
        aTrackLine.erase(0,1);
    } while (aTrackLine.length() == 1);
    // Run evaluation tool -----------------------------------------------------
    std::string s = binaryPath + "MoSegEvalPR ";
    s += aShotLine + ' ' + aTrackLine;
    sprintf(dummy, " %f", objectThreshold);
    s += dummy;
    std::cout << "EXECUTE: " << s.c_str() << std::endl;
    if (system(s.c_str()) != 0) {
      std::cerr << "Error while running " << s << std::endl;
      return -1;
    }
    // Evaluate result file ----------------------------------------------------
    aCounter++;
    s = aTrackLine;
    s.erase(s.find_last_of('.'),s.length());
    std::ifstream aResult((s+"Numbers.txt").c_str());
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    // Number of evaluated frames
    int aEvaluatedFrames;
    aResult.getline(dummy,300);
    aResult >> aEvaluatedFrames; aResult.getline(dummy,300);
    if (aEvaluatedFrames != aEvaluationMode && aEvaluationMode > 0) {
      std::cerr << "The tracks listed in " << args[3] << " have been computed considering different numbers of frames." << std::endl;
      return -1;
    }
    // >>> avergae region Densities <<<
    std::string sdummy = "";
    int error_counter = 0;
    while (sdummy.find("Average region density") == std::string::npos) {
      aResult.getline(dummy,300);
      sdummy = std::string(dummy);
      error_counter ++;
      if (error_counter > 500) { std::cerr << "Could not find \"Average region density\" in the file!" << std::endl; return -1; }
    }
    float aAvrgDensity;
    aResult >> aAvrgDensity; aResult.getline(dummy,300);
    aSumAvrgDensity += aAvrgDensity;
    // >>> Average Precision Recall F-measure <<<
    sdummy = "";
    error_counter = 0;
    while (sdummy.find("Average Precision, Recall, F-measure") == std::string::npos) {
      aResult.getline(dummy,300);
      sdummy = std::string(dummy);
      error_counter ++;
      if (error_counter > 500) { std::cerr << "Could not find \"Average Precision, Recall, F-measure\" in the file!" << std::endl; return -1; }
    }
    float aAvrgPrecision;
    float aAvrgRecall;
    float aThisFmeasure;
    aResult >> aAvrgPrecision >> aAvrgRecall >> aThisFmeasure;
    std::cout << "(P,R)=(" << aAvrgPrecision << "," <<  aAvrgRecall << ") " << std::endl;
    aSumAvrgPrecision += aAvrgPrecision;
    aSumAvrgRecall += aAvrgRecall;
    aSumAvrgPrecisionSq += aAvrgPrecision*aAvrgPrecision;
    aSumAvrgRecallSq += aAvrgRecall*aAvrgRecall;
    // >>> visibile count <<< 
    sdummy = "";
    error_counter = 0;
    while (sdummy.find("Visible objects") == std::string::npos) {
      aResult.getline(dummy,300);
      sdummy = std::string(dummy);
      error_counter ++;
      if (error_counter > 500) { std::cerr << "Could not find \"Visible objects\" in the file!" << std::endl; return -1; }
    }
    int aVisibles;
    aResult >> aVisibles; aResult.getline(dummy,300);
    aSumVisibles += aVisibles;
    // >>> Extracted objects <<<
    sdummy = "";
    error_counter = 0;
    while (sdummy.find("Extracted objects") == std::string::npos) {
      aResult.getline(dummy,300);
      error_counter ++;
      sdummy = std::string(dummy);
      if (error_counter > 500) { std::cerr << "Could not find \"Extracted objects\" in the file!" << std::endl; return -1; }
    }
    int aExtracted;
    aResult >> aExtracted;
    aSumExtracted += aExtracted;
    std::cout << "extracted objects counter: " << aSumExtracted << " / " << aSumVisibles << std::endl;
  }
  float invSize = 1.0f/aCounter;
  aSumAvrgPrecision *= invSize;
  aSumAvrgRecall *= invSize;
  aSumAvrgPrecisionSq *= invSize;
  aSumAvrgRecallSq *= invSize;
  float aStdDevRecall = sqrt(aSumAvrgRecallSq-aSumAvrgRecall*aSumAvrgRecall);
  float aStdDevPrecision = sqrt(aSumAvrgPrecisionSq-aSumAvrgPrecision*aSumAvrgPrecision);
  float aAvrgFmeasure = 2.0f*aSumAvrgPrecision*aSumAvrgRecall/(aSumAvrgPrecision+aSumAvrgRecall);
  
  // Write overall outcome -----------------------------------------------------
  aOut << "Evaluation results for segmentations obtained using ";
  if (aEvaluationMode > 0) aOut << aEvaluationMode;
  else aOut << "all";
  sprintf(dummy, "%.2f", objectThreshold);
  std::string sdummy = dummy;
  aOut << " frames" << std::endl;
  aOut << "MoSegEval Version 2.0" << std::endl << std::endl;
  aOut << aCounter <<  " shots were evaluated." << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Region density (in percent): " << std::endl << aSumAvrgDensity*invSize << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Overall precision:" << std::endl 
       << aSumAvrgPrecision << " (std.dev.: " << aStdDevPrecision << ")" << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Overall recall:" << std::endl 
        << aSumAvrgRecall << " (std.dev.: " << aStdDevRecall << ")" << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Overall F-measure: " << std::endl << aAvrgFmeasure << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Extracted objects (#{F-measure > " << sdummy << "}) (excluding background):" << std::endl;
  aOut << aSumExtracted << std::endl;
  aOut << "of " << aSumVisibles << " visible objects" << std::endl;
  return 0;
}
