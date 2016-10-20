#ifndef Arvis_Tracker_H
#define Arvis_Tracker_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <deque>
#include <tuple>
#include <time.h>	//for testing


using cv::Mat;
using cv::Scalar;
using cv::KeyPoint;
using cv::waitKey;
using cv::Point;
using std::vector;
using std::tuple;
using std::deque;
using std::cout;
using std::cin;
using std::endl;



//----------------------Color library----------------------
const static uchar Color1[20][3] = {{200,0,0},{0,0,200},{0,200,0},{200,200,0},{0,200,200},
{200,0,200},{0,80,255},{220,150,200},{0,250,100},{100,0,250},{100,250,0},{240,240,100},
{180,250,20},{160,190,20},{25,255,155},{180,130,70},{32,165,220},{120,150,255},{125,25,25},
{105,180,180}};

//----------------------Arvis Tracker Model Struct-----------------------
typedef tuple<float, KeyPoint, vector<float>> Desc_tuple;

struct Tracker_struct
{
	vector<int> States;

	deque<Desc_tuple> OROI_desc;

	vector<Desc_tuple> CROI_desc;

	vector<int> Avg_Dpl;
	
	int fr_trck;
	
	int fr_lost;

	Scalar color;
};

//--------------------------Arvis Tracker Class--------------------------
class Arvis_Tracker
{
public:

	//-------------Main Functions-------------
	Arvis_Tracker(Mat iniFr, bool Trck_type_BGS);

	~Arvis_Tracker();

	void Initialize_ROI_User(vector<vector<int>> &Ini_states);

	void Initialization(vector<vector<int>> Ini_states, Mat Fr, Mat Fmask = Mat::ones(0, 0, CV_8UC1));

	void Initialization(vector<vector<int>> Ini_states, vector<vector<vector<float>>> Ini_desc, 
		vector<vector<KeyPoint>> Ini_kpt);

	void Search_and_Match(Mat Fr);

	void Tracking_Plot();

	//---------------Variables----------------
	vector<vector<int>> ini_states;	//Initial states when not given by a detection algorithm

private:

	//-------------Main Functions-------------
	void SROI_Characterization(Mat Fr, vector<int> States, vector<vector<float>> &Desc, 
		vector<KeyPoint> &Kpts);

	void Arrange_descriptors(vector<int> States, vector<vector<float>> &Odesc, 
		vector<KeyPoint> &Okpt, vector<vector<float>> &Cdesc, vector<KeyPoint> &Ckpt, 
		Mat Fmask = Mat::ones(0, 0, CV_8UC1));

	void Match_Descriptors(deque<Desc_tuple> Odesc, vector<Desc_tuple> Cdesc, 
		vector<vector<float>> SDesc, vector<KeyPoint> Skpt, vector<int> &Match_idxs, int &Nmatch);

	void ROI_avg_displ(deque<Desc_tuple> OROI_desc, vector<cv::KeyPoint> SROI_kpt, 
		vector<int> Match_idxs, vector<int> &displ);

	void States_from_descriptors(vector<KeyPoint> SROI_kpt, vector<int> Match_idxs, 
		vector<int> &C_states);

	void Get_CROI_descriptors(vector<int> Cstates, vector<int> Match_idxs, vector<KeyPoint> SROI_kpt, 
		vector<vector<float>> SROI_desc, vector<Desc_tuple> &C_desc, Mat Fmask = Mat::ones(0, 0, CV_8UC1));

	void Updt_Models_Info(vector<int> C_states, vector<Desc_tuple> C_desc, vector<int> Match_idxs, 
		vector<KeyPoint> S_kpt, vector<vector<float>> S_desc, Tracker_struct &Model);

	//-----------Auxiliar Functions-----------
	void detKeys_SURF(const Mat &img, vector<KeyPoint> &keys, vector<vector<float>> &Ini_desc, 
		Mat Fmask = Mat::ones(0, 0, CV_8UC1)); 

	vector<vector<float>> Mat2Vector(Mat A);

	float GKernel_vectors(vector<float> A, vector<float> B, float BW);

	static void MouseHandler(int event, int x, int y, int flags, void* param);

	void Sort_Desc(deque<Desc_tuple> &O_Desc);

	void validate_states(vector<int> &States);

	bool kpt_in_ROI(KeyPoint a, Mat Fmask, float lrg_kpt_in);

	bool kpt_in_ROI(KeyPoint a, vector<int> States, float lrg_kpt_in);

	//-----------Testing Functions-----------
	void PlotKpts(vector<KeyPoint> A, Mat &Im, char *Pltname, Scalar S1 = Scalar(200,0,0));

	void PlotKpts(vector<Desc_tuple> A, Mat &Im, char *Pltname, Scalar S1 = Scalar(200,0,0));

	void PlotKpts(deque<Desc_tuple> A, Mat &Im, char *Pltname, Scalar S1 = Scalar(200,0,0));

	void DrawRects(vector<int> A, vector<int> B, char* plot1, 
		cv::Scalar S1 = cv::Scalar(255,0,0), cv::Scalar S2 = cv::Scalar(0,255,0));
	

	//---------------Variables----------------
	vector<Tracker_struct> Model;	//Tracker model information
	Mat Im_RGB;						//Current frame copy
	int im_r, im_c;					//Image number of rows and columns
	deque<Scalar> Color_Lib;		//Color library used to plot Bbox tracking
	bool flag_ini;

	//---------------Parameters---------------
	int NDesc_O;			//Number of Desc used to model a ROI (object info)
	int NDesc_C;			//Number of Desc used to model a ROI (context info)
	int min_Desc;			//Minimum number of Desc to consider a PROI
	float NDesc_match;		//Percentage of NDesc_O descriptors required to accept a match 
	float w_ini;			//Descriptor initial weight
	float lrg_kpt_in;		//Percentage to enlarge initial ROIs to accept kpts inside them
	int srch_area;			//Area to search and track an object, is given by a percentage of the frame size
	float thr_match;		//Threshold to validate the match between 2 descriptors  
	float thr_ctxt;			//Threshold to validate the match between 2 descriptors (context match) 
	float GKernel_BW;		//Gaussian kernel band width to compare 2 descriptors
	float Desc_srch;		//Distance threshold to accept a descriptor match
	float Rad_thr;			//Radius threshold to accept a descriptor match 
	float C_ROI_lrg;		//Percentage to enlarge the Candidate ROI bounding box
	float States_updt_step;	//Model states updating step
	float w_updt;			//Descriptors weight updating parameter
	float Desc_updt_rate;	//Descriptors updating rate (Percentage of NDesc_O)
};
#endif
