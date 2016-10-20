#include "Arvis_Tracker.h"
#include "Datasets_Menu.h"


void main()
{
	//Load video
	Datasets_Menu M;
	M.Select_Dataset();	

	//Load initial frame
	Mat Fr_o, Fr; 
	M.Get_Frame(Fr_o);

	cv::resize(Fr_o, Fr, cv::Size(640,480));

	//Create tracking object 
	//flag_detection = true: Initialize tracking with states given by an object detection algorithm
	//flag_detection = false: Initialize tracking with states given by bbox given by user
	bool flag_detection = false;
	Arvis_Tracker A_tracker(Fr, flag_detection);

	//Initialize tracking
	A_tracker.Initialization(A_tracker.ini_states,Fr);

	int fr_idx = 0;

	for(;;)
	{
		M.Get_Frame(Fr_o);
		cv::resize(Fr_o, Fr, cv::Size(640,480));

		if(Fr.empty())
			break;

		A_tracker.Search_and_Match(Fr);
		A_tracker.Tracking_Plot();

		fr_idx++;

		//cv::imshow("Frame",Fr);
		cv::waitKey(1);
	}
}