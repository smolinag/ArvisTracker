#include "Arvis_Tracker.h"

//-------------------------------------------------------------------------------------
//Global Variables
Mat Frame, Frame_1;
int drag = 0;
cv::Point point1, point2;
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
//Constructor
//-------------------------------------------------------------------------------------
Arvis_Tracker::Arvis_Tracker(Mat iniFr, bool Trck_type_BGS)
{
	Frame = iniFr;

	im_r = iniFr.rows;
	im_c = iniFr.cols;

	Desc_srch = float(im_r)*0.1; //set as 15% the frame height (If the FPS is known set according it)

	Rad_thr = 30;
	
	NDesc_O = 70;

	NDesc_C = 50;

	NDesc_match = 0.1;

	min_Desc = int(ceil(float(NDesc_O)*0.1));

	w_ini = float(0.1);

	lrg_kpt_in = 0.2;

	srch_area = float(im_r)*0.05;	//set as 5% the frame height (If the FPS is known set according it)

	GKernel_BW = 0.7;

	thr_match = 0.8;

	thr_ctxt = 0.75;

	C_ROI_lrg = 0;

	States_updt_step = 0.15;

	w_updt = 0.1;

	Desc_updt_rate = int(ceil(float(NDesc_O)*0.15));

	//Copy color library
	for(int i=0; i<20; i++)
	{
		Color_Lib.push_back(Scalar(Color1[i][0],Color1[i][1],Color1[i][2]));
	}

	//Initialize tracking states with rectangle drew by the user	
	if(!Trck_type_BGS)
	{
		Initialize_ROI_User(ini_states);
	}
}


Arvis_Tracker::~Arvis_Tracker()
{

}

//-------------------------------------------------------------------------------------
//Initialize Bbox states by input rectangles from user
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Initialize_ROI_User(vector<vector<int>> &Ini_states)
{
	char* src_window = "Frame";	
	cv::namedWindow(src_window, 0 );
	flag_ini = false;
	Frame_1 = Frame.clone();

	vector<int> aux_ini_states;

	while(!flag_ini)
	{		
		cv::imshow(src_window,Frame_1);
		cv::waitKey(10);
		cv::setMouseCallback(src_window, MouseHandler, &flag_ini);
	}

	//Get states validating points coordinates
	int x, y, wid, heig, ctrx, ctry;

	if(point1.x<point2.x)
	{
		x = point1.x;
		wid = point2.x-point1.x;
	}
	else
	{
		x = point2.x;
		wid = point1.x-point2.x;
	}
	
	if(point1.y<point2.y)
	{
		y = point1.y;
		heig = point2.y-point1.y;
	}
	else
	{
		y = point2.y;
		heig = point1.y-point2.y;
	}

	ctrx = x+int(floor(double(wid)/2.));
	ctry = y+int(floor(double(heig)/2.));

	//store states
	aux_ini_states.push_back(x);
	aux_ini_states.push_back(y);
	aux_ini_states.push_back(wid);
	aux_ini_states.push_back(heig);
	aux_ini_states.push_back(ctrx);
	aux_ini_states.push_back(ctry);
	
	//store in global memory
	Ini_states.push_back(aux_ini_states);
}

//-------------------------------------------------------------------------------------
//Initialize tracker models with given states and given descriptors information bbox user
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Initialization(vector<vector<int>> Ini_states,  Mat Fr, Mat Fmask)
{
	Tracker_struct New_MROI;
	int NDesc_ROI_O, NDesc_ROI_C, rdm_n;
	deque<Desc_tuple> Aux_desc_O;
	vector<Desc_tuple> Aux_desc_C;

	//Loop through Initialization ROIs
	for (int i=0; i<Ini_states.size(); i++)
	{	
		//Characterize search regions around given states
		vector<vector<float>> O_desc, C_desc;
		vector<KeyPoint> O_kpt, C_kpt;
		SROI_Characterization(Fr, Ini_states[i], O_desc, O_kpt);

		//Arrange descriptors into object info and context info
		Arrange_descriptors(Ini_states[i], O_desc, O_kpt, C_desc, C_kpt);

		//test
		Mat Fr_copy = Fr.clone();
		PlotKpts(O_kpt, Fr_copy, "Kpts");
		//PlotKpts(C_kpt, Fr_copy, "Kpts", Scalar(0,200,0));

		NDesc_ROI_O = O_desc.size();

		//If the initial descriptors size for ROI i is enough initialize new model
		if(NDesc_ROI_O>=min_Desc)
		{
			//Assign states
			New_MROI.States = Ini_states[i];

			//-----Store Object Kpts and Descriptors-----
			
			//Initialize auxiliar descriptor vector
			Aux_desc_O.clear();

			//If there are more descriptors than the required -> randomly subsample, else -> fill the rest with zeros
			if(NDesc_ROI_O>NDesc_O)
			{
				for (int j=0; j<NDesc_O; j++)
				{
					//Get random index
					NDesc_ROI_O = O_desc.size();
					rdm_n = rand() % NDesc_ROI_O;	

					//Fill with random descriptor from Query
					Aux_desc_O.push_back(std::make_tuple(w_ini, O_kpt[rdm_n], O_desc[rdm_n]));	

					//Delete used descriptor information
					O_desc.erase(O_desc.begin()+rdm_n);	
					O_kpt.erase(O_kpt.begin()+rdm_n);	
				}
			}
			else
			{
				//Loop through Query Descriptors
				for (int j=0; j<NDesc_ROI_O; j++)	
				{		
					//Fill vector with available Descriptors
					Aux_desc_O.push_back(std::make_tuple(w_ini, O_kpt[j], O_desc[j]));		
				}
			}

			//Assign Descriptor information
			New_MROI.OROI_desc = Aux_desc_O;
			//-----------------------------------------
			
			//-----Store Context Kpts and Descriptors-----
			NDesc_ROI_C = C_desc.size();
			
			//Initialize auxiliar descriptor vector
			Aux_desc_C.clear();

			//If there are more descriptors than the required -> randomly subsample, else -> fill the rest with zeros
			if(NDesc_ROI_C>NDesc_C)
			{
				for (int j=0; j<NDesc_C; j++)
				{
					//Get random index
					NDesc_ROI_C = C_desc.size();
					rdm_n = rand() % NDesc_ROI_C;	

					//Fill with random descriptor from Query
					Aux_desc_C.push_back(std::make_tuple(w_ini, C_kpt[rdm_n], C_desc[rdm_n]));	

					//Delete used descriptor information
					C_desc.erase(C_desc.begin()+rdm_n);	
					C_kpt.erase(C_kpt.begin()+rdm_n);	
				}
			}
			else
			{
				//Loop through Query Descriptors
				for (int j=0; j<NDesc_ROI_C; j++)	
				{		
					//Fill vector with available Descriptors
					Aux_desc_C.push_back(std::make_tuple(w_ini, C_kpt[j], C_desc[j]));		
				}
			}

			//Assign Descriptor information
			New_MROI.CROI_desc = Aux_desc_C;
			//-----------------------------------------
			
			//Assign number of tracked frames
			New_MROI.fr_trck = 1;

			//Assign number of lost frames
			New_MROI.fr_trck = 0;

			//Assign number of tracked frames
			New_MROI.color = Color_Lib[0];

			//Delete assigned color from color library
			Color_Lib.pop_front();

			//Store new modeled ROI in global memory
			Model.push_back(New_MROI);
		}
	}
}

//-------------------------------------------------------------------------------------
//Initialize tracker models with given states and given descriptors information
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Initialization(vector<vector<int>> Ini_states, vector<vector<vector<float>>> Ini_desc, 
		vector<vector<KeyPoint>> Ini_kpt)
{
	Tracker_struct New_MROI;

	int NDesc_ROI, rdm_n;

	deque<Desc_tuple> Aux_desc_tuple;

	//Loop through Initialization ROIs
	for (int i=0; i<Ini_states.size(); i++)
	{	
		NDesc_ROI = Ini_desc[i].size();

		//If the initial descriptors size for ROI i is enough initialize new model
		if(NDesc_ROI>=min_Desc)
		{
			//Assign states
			New_MROI.States = Ini_states[i];

			//Initialize auxiliar descriptor vector
			Aux_desc_tuple.clear();

			//If there are more descriptor than the required -> randomly subsample, else -> fill the rest with zeros
			if(NDesc_ROI>NDesc_O)
			{
				for (int j=0; j<NDesc_O; j++)
				{
					//Get random index
					rdm_n = rand() % NDesc_ROI;	

					//Fill with random descriptor from Query
					Aux_desc_tuple.push_back(std::make_tuple(w_ini, Ini_kpt[i][rdm_n], Ini_desc[i][rdm_n]));	

					//Delete used descriptor information
					Ini_desc[i].erase(Ini_desc[i].begin()+rdm_n);	
					Ini_kpt[i].erase(Ini_kpt[i].begin()+rdm_n);	
				}
			}
			else
			{
				//Loop through Query Descriptors
				for (int j=0; j<NDesc_ROI; j++)	
				{		
					//Fill vector with available Descriptors
					Aux_desc_tuple.push_back(std::make_tuple(w_ini, Ini_kpt[i][rdm_n], Ini_desc[i][rdm_n]));		
				}
			}

			//Assign Descriptor information
			New_MROI.OROI_desc = Aux_desc_tuple;

			//Assign number of tracked frames
			New_MROI.fr_trck = 1;

			//Assign number of lost frames
			New_MROI.fr_trck = 0;

			//Assign number of tracked frames
			New_MROI.color = Color_Lib[0];

			//Delete assigned color from color library
			Color_Lib.pop_front();

			//Store new modeled ROI in global memory
			Model.push_back(New_MROI);
		}
	}
}

//-------------------------------------------------------------------------------------
//Characterize search regions around given bounding boxes
//-------------------------------------------------------------------------------------
void Arvis_Tracker::SROI_Characterization(Mat Fr, vector<int> States, vector<vector<float>> &Desc, 
		vector<KeyPoint> &Kpts)
{
		//Get search region
		States[0] = States[0]-srch_area;
		States[1] = States[1]-srch_area;
		States[2] = States[2]+2*srch_area;
		States[3] = States[3]+2*srch_area;

		//Validate search ROI states 
		validate_states(States);

		//Extract Smask image
		Mat Smask = Mat::zeros(im_r,im_c,CV_8UC1);
		Mat Aux(Smask, cv::Rect(States[0], States[1], States[2], States[3]));
		Aux = Scalar(255);

		//Get Keypoints and Descriptors
		detKeys_SURF(Fr, Kpts, Desc, Smask);
}

//-------------------------------------------------------------------------------------
//Initial arrange descriptors into context or object features depending on the ROI given by BGS 
//and the object states
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Arrange_descriptors(vector<int> States, vector<vector<float>> &Odesc, 
		vector<KeyPoint> &Okpt, vector<vector<float>> &Cdesc, vector<KeyPoint> &Ckpt, Mat Fmask)
{
	bool in_ROI, in_Mask;

	Cdesc = Odesc;
	Ckpt = Okpt;

	vector<KeyPoint> Aux = Ckpt;
	
	int jo=0, jc=0;

	//Loop through Keypoints
	for(int j=0; j<Aux.size(); j++)
	{
		in_ROI = kpt_in_ROI(Aux[j], States, lrg_kpt_in);

		if(Fmask.rows>0)
			in_Mask = kpt_in_ROI(Aux[j], Fmask, lrg_kpt_in);
		else
			in_Mask = true;

		//Check if Keypoint is inside bbox and mask to store in object info, else store in context info
		if(in_ROI&&in_Mask)
		{
			Cdesc.erase(Cdesc.begin()+jc);
			Ckpt.erase(Ckpt.begin()+jc);
			jo++;
		}
		else
		{
			Odesc.erase(Odesc.begin()+jo);
			Okpt.erase(Okpt.begin()+jo);
			jc++;
		}
	}
}

//-------------------------------------------------------------------------------------
//Given a Set of object models, search in search regions the best matching descriptors
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Search_and_Match(Mat Fr)
{
	Im_RGB = Fr.clone();

	for(int i=0; i<Model.size(); i++)
	{
		//Characterize search regions around given states
		vector<vector<float>> S_desc;
		vector<KeyPoint> S_kpt;
		SROI_Characterization(Fr, Model[i].States, S_desc, S_kpt);
		//Mat Skpts = Fr.clone();
		//PlotKpts(S_kpt, Skpts, "Skpts", Scalar(0,0,200));

		//Match OROI descriptors with SROI descriptors
		vector<int> Match_idxs;
		
		//clock_t ini = clock();
		int Nmatch;
		Match_Descriptors(Model[i].OROI_desc, Model[i].CROI_desc, S_desc, S_kpt, Match_idxs, Nmatch);
		//printf("tiempo de calculo: %f seg \n", ((double)clock() - ini)/CLOCKS_PER_SEC);
		//waitKey();

		//Accept new tracked region if at least thr_desc descriptors matched
		int thr_desc = floor(NDesc_match*double(Model[i].OROI_desc.size()));
		if(Nmatch>thr_desc)
		{
			//Get ROI avg displacement according to matched descriptors
			vector<int> avg_dpl;
			ROI_avg_displ(Model[i].OROI_desc, S_kpt, Match_idxs, avg_dpl);
			//Assign displacement
			Model[i].Avg_Dpl = avg_dpl;

			//Get candidate ROI states according to matched descriptors
			vector<int> Cstates;
			States_from_descriptors(S_kpt, Match_idxs, Cstates);
			//DrawRects(Model[i].States, Cstates, "Bboxes");

			//Get candidate ROI states according to matched descriptors
			vector<Desc_tuple> Cdesc;
			Get_CROI_descriptors(Cstates, Match_idxs, S_kpt, S_desc, Cdesc);
			Mat NewDesc = Fr.clone();
			PlotKpts(Cdesc, NewDesc, "NewDesc", Scalar(0,200,200));

			//Update tracking models
			Updt_Models_Info(Cstates, Cdesc, Match_idxs, S_kpt, S_desc, Model[i]);
		}
		//Check if the region is lost
		else
		{
			//Increase number of lost frames
			Model[i].fr_lost++;
		}
	}
}

//-------------------------------------------------------------------------------------
//Match OROI descriptors with SROI descriptors and see that the matched SROI 
//descriptor doesnt match a CROI descriptor
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Match_Descriptors(deque<Desc_tuple> Odesc, vector<Desc_tuple> Cdesc, 
	vector<vector<float>> SDesc, vector<KeyPoint> Skpt, vector<int> &Match_idxs, int &Nmatch)
{
	float valO, valC, val_ref, desc_dist;
	int j, Match_idx, rad_dif;
	bool match;

	Nmatch = 0;

	//For each OROI descriptor store index of matched SROI descriptor (if found)
	vector<int> Aux (Odesc.size(),-1);	
	Match_idxs = Aux;

	cout<<"Desc:"<<Odesc.size()<<endl;

	//Loop through OROI descriptors
	for(int i=0; i<Odesc.size(); i++)
	{
		val_ref = 0;
		match = false;

		//Loop through SROI descriptors
		for(int j=0; j<SDesc.size(); j++)
		{
			//check if the current index j has already taken 
			std::vector<int>::iterator it = std::find (Match_idxs.begin(), Match_idxs.end(), j);	

			if(it == Match_idxs.end())
			{
				//Distance between OROI i and SROI j centroids
				desc_dist = sqrt(pow(std::get<1>(Odesc[i]).pt.x-Skpt[j].pt.x,2)+pow(std::get<1>(Odesc[i]).pt.y-Skpt[j].pt.y,2));
				
				//Check if descriptors are close enough
				if(desc_dist<Desc_srch) 
				{
					//Radius difference
					rad_dif = abs(std::get<1>(Odesc[i]).size-Skpt[j].size);

					//Check if descriptors radius is similar
					if(rad_dif<Rad_thr)
					{
						//Compute Gkernel between OROI and SROI descriptor
						valO = GKernel_vectors(std::get<2>(Odesc[i]),SDesc[j], GKernel_BW);

						//If current max value and the threshold is surpassed assign match
						if(valO>thr_match&&valO>val_ref)
						{
							Match_idxs[i] = j;	//assign SROI j descriptor index to Model object i
							Match_idx = j;			
							match = true;			//match found true
						}
					}
				}
			}
		}
		//Search if the matched SROI descriptor doesnt match a CROI descriptor 
		if(match)
		{
			//Loop through CROI descriptors
			for(int k=0; k<Cdesc.size(); k++)
			{
				//Radius difference
				rad_dif = abs(std::get<1>(Cdesc[k]).size-Skpt[Match_idx].size);

				//Check if descriptors radius is similar
				if(rad_dif<Rad_thr)
				{
					//Compute Gkernel between SROI and CROI descriptor
					valC = GKernel_vectors(SDesc[Match_idx],std::get<2>(Cdesc[k]), GKernel_BW);

					//If the Gkernel value exceeds a threshold delete match and label OROI with -2
					//to delete it because its not distinctive
					if(valC>thr_ctxt)
					{
						match = false;
						Match_idxs[i] = -2;

						////Test non Relevant descriptors
						//vector<KeyPoint> NonRel1, NonRel2, NonRel3;
						//NonRel1.push_back(std::get<1>(Odesc[i]));
						//NonRel2.push_back(Skpt[Match_idx]);
						//NonRel3.push_back(std::get<1>(Cdesc[k]));
						//Mat NonRelIm = Im_RGB.clone();
						//PlotKpts(NonRel1, NonRelIm, "NonRel", cv::Scalar(200,250,0));
						//PlotKpts(NonRel2, NonRelIm, "NonRel", cv::Scalar(0,200,0));
						//PlotKpts(NonRel3, NonRelIm, "NonRel", cv::Scalar(200,0,0));
						////-----------------------------
					}
					else
						Nmatch++;
				}
			}
		}
	}
}

//-------------------------------------------------------------------------------------
//From matched STIPs get the average displacement of MROI M_idx
//-------------------------------------------------------------------------------------
void Arvis_Tracker::ROI_avg_displ(deque<Desc_tuple> OROI_desc, vector<KeyPoint> SROI, 
	vector<int> Match_idxs, vector<int> &displ)
{
	double dispx = 0, dispy = 0;
	int s = 0;

	vector<cv::KeyPoint> TestV;

	for(int i=0; i<Match_idxs.size(); i++)
	{
		if((Match_idxs[i]!=-1)&&(Match_idxs[i]!=-2))
		{
			s++;
			dispx = double(SROI[Match_idxs[i]].pt.x-std::get<1>(OROI_desc[i]).pt.x) + dispx;
			dispy = double(SROI[Match_idxs[i]].pt.y-std::get<1>(OROI_desc[i]).pt.y) + dispy;
			TestV.push_back(SROI[Match_idxs[i]]);
		}
	}
	displ.push_back(int(floor(dispx/s)));
	displ.push_back(int(floor(dispy/s)));

	PlotKpts(TestV, Im_RGB, "KptsMatched", cv::Scalar(200,0,100));
}

//-------------------------------------------------------------------------------------
//Get candidate ROI states from tracked descriptors
//-------------------------------------------------------------------------------------
void Arvis_Tracker::States_from_descriptors(vector<KeyPoint> SROI, vector<int> Match_idxs, 
		vector<int> &Cstates)
{
	//Vectors to store the x and y coordinates of the matched descriptors
	vector<int> Kpts_x;
	vector<int> Kpts_y;

	int rad;
	
	//Loop through Model matches vector
	for(int i=0; i<Match_idxs.size(); i++)
	{
		//If for Model ROI i there is a Query STIP match, store the Query Stip coordinates
		if((Match_idxs[i]!=-1)&&(Match_idxs[i]!=-2))
		{
			rad = int(floor(double(SROI[Match_idxs[i]].size)/2.));

			Kpts_x.push_back(SROI[Match_idxs[i]].pt.x-rad);
			Kpts_x.push_back(SROI[Match_idxs[i]].pt.x+rad);
			Kpts_y.push_back(SROI[Match_idxs[i]].pt.y-rad);
			Kpts_y.push_back(SROI[Match_idxs[i]].pt.y+rad);
		}
	}

	int min_x, max_x, min_y, max_y, wid, heig, aux, ctr_x, ctr_y;

	//Find Candidate ROI states according to matched Query STIP coordinates---------
	min_x = *std::min_element(Kpts_x.begin(),Kpts_x.end());   
	max_x = *std::max_element(Kpts_x.begin(),Kpts_x.end());
	min_y = *std::min_element(Kpts_y.begin(),Kpts_y.end());
	max_y = *std::max_element(Kpts_y.begin(),Kpts_y.end());
	wid = max_x-min_x;
	heig = max_y-min_y;

	//Enlarge Candidate ROI bounding box---
	min_x = min_x - int(ceil(C_ROI_lrg*wid));
	min_y = min_y - int(ceil(C_ROI_lrg*heig));
	wid = wid + int(ceil(2*C_ROI_lrg*wid));
	heig = heig + int(ceil(2*C_ROI_lrg*heig));

	Cstates.push_back(min_x);
	Cstates.push_back(min_y);
	Cstates.push_back(wid);
	Cstates.push_back(heig);

	validate_states(Cstates);	//validate states

	ctr_x = min_x+int(floor(double(Cstates[2])/2.));
	ctr_y = min_y+int(floor(double(Cstates[3])/2.));

	Cstates.push_back(ctr_x);
	Cstates.push_back(ctr_y);
}

//-------------------------------------------------------------------------------------
//Get new candidate descriptors form CROI to be included into the model
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Get_CROI_descriptors(vector<int> Cstates, vector<int> Match_idxs, 
	vector<KeyPoint> SROI_kpt, vector<vector<float>> SROI_desc, vector<Desc_tuple> &C_desc, Mat Fmask)
{
	bool inmask;

	//For each descriptor that didn´t match see if it is inside Cstates and inside Fmask (if any)
	for(int i=0; i<SROI_kpt.size(); i++)
	{
		if(kpt_in_ROI(SROI_kpt[i], Cstates, 0))	
		{
			if(Fmask.rows==0)
				inmask = true;
			else
				inmask = kpt_in_ROI(SROI_kpt[i], Fmask, 0);

			if(inmask)
			{
				//Store in C_desc only the descriptors that didn't match any Model descriptor but are within Cstates,
				std::vector<int>::iterator it = std::find (Match_idxs.begin(), Match_idxs.end(), i);	

				if (it == Match_idxs.end())
				{
					//Make candidate descriptors
					C_desc.push_back(std::make_tuple(w_ini, SROI_kpt[i], SROI_desc[i]));
				}
			}
		}
	}
}

//-------------------------------------------------------------------------------------
//Update Model ROI states and descriptors according to Candidate ROI
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Updt_Models_Info(vector<int> C_states, vector<Desc_tuple> C_desc, vector<int> Match_idxs, 
		vector<KeyPoint> S_kpt, vector<vector<float>> S_desc, Tracker_struct &Model)
{
	//------------------------------Update states-----------------------------
	Model.States[2] = States_updt_step*C_states[2]+(1-States_updt_step)*Model.States[2];
	Model.States[3] = States_updt_step*C_states[3]+(1-States_updt_step)*Model.States[3];
	Model.States[0] = C_states[4]-int(floor(float(Model.States[2])/2.));
	Model.States[1] = C_states[5]-int(floor(float(Model.States[3])/2.));
	Model.States[4] = C_states[4];
	Model.States[5] = C_states[5];
	//Validate states
	validate_states(Model.States);
	
	//---------------------Update frames tracked and lost---------------------
	Model.fr_trck = Model.fr_trck+1;
	Model.fr_lost = 0;

	int Nm = 0, Nr = 0, No_m = 0;		//test
		
	//------------Update matched Descriptors weights and coordinates----------
	for(int i=0; i<Model.OROI_desc.size();i++)
	{
		if((Match_idxs[i]!=-1)&&(Match_idxs[i]!=-2))
		{
			//weights
			std::get<0>(Model.OROI_desc[i]) += w_updt;

			//coordinates
			std::get<1>(Model.OROI_desc[i]) = S_kpt[Match_idxs[i]];

			//HOG
			std::get<2>(Model.OROI_desc[i]) = S_desc[Match_idxs[i]];

			Nm++;	//test
		}
	}

	//------------Delete Descriptors that also matched with Context----------
	for(int i=0; i<Model.OROI_desc.size();i++)
	{
		if(Match_idxs[i]==-2)
		{
			Model.OROI_desc.erase(Model.OROI_desc.begin()+i);
			
			Nr++;		//test
		}
	}

	//----------Displace non-matched Descriptors according avg displ---------
	int dx = Model.Avg_Dpl[0];
	int dy = Model.Avg_Dpl[1];

	for(int i=0; i<Model.OROI_desc.size();i++)
	{
		if(Match_idxs[i]==-1)
		{
			std::get<1>(Model.OROI_desc[i]).pt.x += dx; 
			std::get<1>(Model.OROI_desc[i]).pt.y += dy; 
			No_m++;		//test
		}
	}

	//------------Sort Model descriptors according to its weights------------
	Sort_Desc(Model.OROI_desc);
	//Mat Upd = Im_RGB.clone();
	//PlotKpts(Model.OROI_desc, Upd, "Kpts_Upd", cv::Scalar(255,0,0));

	//------------------------Assign new Descriptors---------------------------
	int Desc_replaced = 0, rdm_n;

	while(C_desc.size()>0)
	{
		rdm_n = rand() % C_desc.size();	//get random index

		//If the Model has all the NStips_ROI STIPs, randomly replace STIP_updt_rate STIPs with the lowest weight with Candidate STIPs
		if(Model.OROI_desc.size()==NDesc_O)
		{
			if(Desc_replaced<=Desc_updt_rate)
			{
				Model.OROI_desc.pop_back();					//delete Model STIP with the lowest weight
				Model.OROI_desc.push_front(C_desc[rdm_n]);	//store Candidate STIP
				C_desc.erase(C_desc.begin()+rdm_n);			//delete used Candidate STIP
				Desc_replaced++;							//Increment number of STIPs replaced
			}
			else
			{
				break;
			}			
		}
		else	//Else finish filling the Model NStips_ROI STIPs
		{	
			Model.OROI_desc.push_front(C_desc[rdm_n]);	//store Candidate STIP
			C_desc.erase(C_desc.begin()+rdm_n);			//delete used Candidate STIP
			Desc_replaced++;							//Increment number of STIPs replaced
		}
	}

	cout<<"IniM:"<<Match_idxs.size()<<" Matches:"<<Nm<<" No_M:"<<No_m<<" No_Rel:"<<Nr<<" New:"<<Desc_replaced<<" UpdM:"<<Model.OROI_desc.size()<<endl;
	Mat Upd = Im_RGB.clone();
	PlotKpts(Model.OROI_desc, Upd, "Kpts_Upd", cv::Scalar(255,0,0));
}

//-------------------------------------------------------------------------------------
//Get SURF characterization 
//-------------------------------------------------------------------------------------
void Arvis_Tracker::detKeys_SURF(const Mat &img, vector<KeyPoint> &keys, vector<vector<float>> &Desc, Mat Fmask) 
{
	double hessThresh = 400.0;  // por defecto 400;
	int nOctaves = 3;  // por defecto 3
	int nLayers = 2;
	bool extended = true;
	bool upright = true;  // por defecto true
	Mat Desc_Mat;

	//Set detector
	cv::SURF detector(hessThresh, nOctaves, nLayers, extended, upright);

	//Set ROI mask for detector
	if(Fmask.rows == 0)
		Fmask = Mat::ones(img.rows, img.cols, CV_8UC1);

	//Detect keypoints
	detector(img,Fmask,keys);

	//Compute descriptors
	detector.compute(img, keys, Desc_Mat);

	//Convert descriptors to vector
	Desc = Mat2Vector(Desc_Mat);
}

//-------------------------------------------------------------------------------------
//Validate if generated states are inside the image
//-------------------------------------------------------------------------------------
void Arvis_Tracker::validate_states(vector<int> &S)
{
	if(S[0]<0)
	{
		S[2] = S[2]-S[0];
		S[0] = 0;
	}

	if(S[1]<0)
	{
		S[3] = S[3]-S[1];
		S[1] = 0;
	}

	if((S[0]+S[2])>im_c)
		S[2] = im_c-S[0]-1;

	if((S[1]+S[3])>im_r)
		S[3] = im_r-S[1]-1;
}

//-------------------------------------------------------------------------------------
//Convert Mat to vector
//-------------------------------------------------------------------------------------
vector<vector<float>> Arvis_Tracker::Mat2Vector(Mat A)
{
	vector<vector<float>> B;
	vector<float> aux;

	if(A.size>0)
	{
		for(int i=0; i<A.rows; i++)
		{
			aux.clear();
			for(int j=0; j<A.cols; j++)
			{
				aux.push_back( A.at<float>(i,j));
			}
			B.push_back(aux);
		}
	}
	else
		cout<<"Sample size must be >0";

	return B;
}

//-------------------------------------------------------------------------------------
//Calculate the 2-norm of the difference of vectors <A-B>
//-------------------------------------------------------------------------------------
float Arvis_Tracker::GKernel_vectors(vector<float> A, vector<float> B, float BW)
{
	float normv = 0, Gkernel;

	for(int i=0; i<A.size(); i++)
	{
		normv = (A[i]-B[i])*(A[i]-B[i])+normv;
	}	
	
	Gkernel = exp(-normv/(2*BW*BW));

	return Gkernel;
}

//-------------------------------------------------------------------------------------
//Sort Model descriptors according to their weights
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Sort_Desc(deque<Desc_tuple> &O_Desc)
{
		std::sort(O_Desc.begin(),O_Desc.end(),
			[](const  Desc_tuple& a,
			const  Desc_tuple& b) -> bool
		{
			return std::get<0>(a) > std::get<0>(b);
		});
}

//-------------------------------------------------------------------------------------
//Draw rectangle with mouse to initialize traking states
//-------------------------------------------------------------------------------------
void Arvis_Tracker::MouseHandler(int event, int x, int y, int flags, void* param)
{
	bool* flag = (bool*) param;

    if (event == CV_EVENT_LBUTTONDOWN && !drag && !(*flag))
    {
        /* left button clicked. ROI selection begins */
        point1 = cv::Point(x, y);
        drag = 1;
    }

	else if (event == CV_EVENT_MOUSEMOVE && drag && !(*flag))
    {
        /* mouse dragged. ROI being selected */
        Frame_1 = Frame.clone();
        point2 = cv::Point(x, y);
        cv::rectangle(Frame_1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        cv::imshow("Frame", Frame_1);
    }

    else if (event == CV_EVENT_LBUTTONUP && drag && !(*flag))
    {
        Frame_1 = Frame.clone();
        point2 = cv::Point(x, y);
        drag = 0;
        (*flag) = true;
        cv::imshow("Frame", Frame_1);
    }
}

//-------------------------------------------------------------------------------------
//Look if a keypoint is enclosed in a ROI by checking if the four cardinal points given 
//by the keypoint centroid and radius are inside the ROI
//-------------------------------------------------------------------------------------
bool Arvis_Tracker::kpt_in_ROI(cv::KeyPoint a, vector<int> States, float lrg_kpt_in)
{
	int cx, cy;

	bool flagin_l, flagin_r, flagin_u, flagin_d, flagin;

	//left
	cx = a.pt.x-floor((1-lrg_kpt_in)*a.size/2);
	if(cx<0)
		cx = 0;
	flagin_l = cx>States[0];

	//right
	cx = a.pt.x+floor((1-lrg_kpt_in)*a.size/2);
	if(cx>=im_c)
		cx = im_c-1;
	flagin_r = cx<(States[0]+States[2]);

	//up
	cy = a.pt.y-floor((1-lrg_kpt_in)*a.size/2);
	if(cy<0)
		cy = 0;
	flagin_u = cy>States[1];

	//down
	cy = a.pt.y+floor((1-lrg_kpt_in)*a.size/2);
	if(cy>=im_r)
		cy = im_r+1;
	flagin_d = cy<(States[1]+States[3]);

	flagin = flagin_l&&flagin_r&&flagin_u&&flagin_d;

	return flagin;
}

//-------------------------------------------------------------------------------------
//Look if a keypoint is enclosed in a ROI by checking if the four cardinal points given 
//by the keypoint centroid and radius are inside the ROI
//-------------------------------------------------------------------------------------
bool Arvis_Tracker::kpt_in_ROI(cv::KeyPoint a, Mat Fmask, float lrg_kpt_in)
{
	int cx, cy;

	bool flagin_l, flagin_r, flagin_u, flagin_d, flagin;

	//left
	cx = a.pt.x-floor((1-lrg_kpt_in)*a.size/2);
	cy = a.pt.y;
	if(cx<0)
		cx = 0;
	flagin_l = (Fmask.at<uchar>(Point(cx,cy))==255);

	//right
	cx = a.pt.x+floor((1-lrg_kpt_in)*a.size/2);
	cy = a.pt.y;
	if(cx>=im_c)
		cx = im_c+1;
	flagin_r = (Fmask.at<uchar>(Point(cx,cy))==255);

	//up
	cx = a.pt.x;
	cy = a.pt.y-floor((1-lrg_kpt_in)*a.size/2);
	if(cy<0)
		cy = 0;
	flagin_u = (Fmask.at<uchar>(Point(cx,cy))==255);

	//down
	cx = a.pt.x;
	cy = a.pt.y+floor((1-lrg_kpt_in)*a.size/2);
	if(cy>=im_r)
		cy = im_r+1;
	flagin_d = (Fmask.at<uchar>(Point(cx,cy))==255);

	flagin = flagin_l&&flagin_r&&flagin_u&&flagin_d;

	return flagin;
}

//-------------------------------------------------------------------------------------
//Plot rectangles for the Model ROIs tracked
//-------------------------------------------------------------------------------------
void Arvis_Tracker::Tracking_Plot()
{
	Mat Im1 = Im_RGB.clone();

	cv::Point ul;
	cv::Point lr;

	char id[5];

	for(int i=0;i<Model.size();i++)
	{
		if(Model[i].fr_lost==0)
		{
			ul = cv::Point(Model[i].States[0],Model[i].States[1]);
			lr = cv::Point(Model[i].States[0]+Model[i].States[2],Model[i].States[1]+Model[i].States[3]);

			cv::rectangle(Im1,ul,lr,Model[i].color,2);

			itoa(i,id,10);

			cv::putText(Im1,id,Point(ul.x+5,ul.y+10),cv::FONT_HERSHEY_SIMPLEX,0.5,Model[i].color,1.5);
		}
	}

	imshow("Tracked Regions:", Im1);
	waitKey();
}

//-------------------------------------------------------------------------------------
//Plot keypoints in image
//-------------------------------------------------------------------------------------
void Arvis_Tracker::PlotKpts(vector<cv::KeyPoint> A, Mat &Im, char *Pltname, Scalar S1)
{
	for(int i=0; i<A.size(); i++)
	{
		circle(Im, A[i].pt, (int)A[i].size/2, S1, 2);
	}

	imshow(Pltname, Im);
	waitKey();
}

//-------------------------------------------------------------------------------------
//Plot keypoints in image
//-------------------------------------------------------------------------------------
void Arvis_Tracker::PlotKpts(vector<Desc_tuple> A, Mat &Im, char *Pltname, Scalar S1)
{
	Mat B = Frame.clone();

	for(int i=0; i<A.size(); i++)
	{
		circle(B, std::get<1>(A[i]).pt, (int)std::get<1>(A[i]).size/2, S1, 2);
	}

	imshow(Pltname, B);
	waitKey();
}

//-------------------------------------------------------------------------------------
//Plot keypoints in image
//-------------------------------------------------------------------------------------
void Arvis_Tracker::PlotKpts(deque<Desc_tuple> A, Mat &Im, char *Pltname, Scalar S1)
{
	Mat B = Frame.clone();

	for(int i=0; i<A.size(); i++)
	{
		circle(B, std::get<1>(A[i]).pt, (int)std::get<1>(A[i]).size/2, S1, 2);
	}

	imshow(Pltname, B);
	waitKey();
}

//-------------------------------------------------------------------------------------
//Draw rectangle A and B in orginal image
//-------------------------------------------------------------------------------------
void Arvis_Tracker::DrawRects(vector<int> A, vector<int> B, char* plot1, cv::Scalar S1, cv::Scalar S2)
{
	Mat Im1 = Im_RGB.clone();
	cv::rectangle(Im1,cv::Point(A[0],A[1]),cv::Point(A[0]+A[2],A[1]+A[3]),S1,2);

	cv::rectangle(Im1,cv::Point(B[0],B[1]),cv::Point(B[0]+B[2],B[1]+B[3]),S2,2);

	if(A.size()>4&&B.size()>4)
	{
		cv::circle(Im1,cv::Point(A[4],A[5]),2,S1,2);
		cv::circle(Im1,cv::Point(B[4],B[5]),2,S2,2);
	}
	
	imshow(plot1,Im1);
	cv::waitKey();
}