#include "Datasets_Menu.h"

//-----------------------------------------------------------------------------------
//Class constructor
Datasets_Menu::Datasets_Menu()
{
	//Path for server databases directory
	main_path_str = "E:\\Databases\\";

	//Image formats
	img_fmt.push_back("png");
	img_fmt.push_back("jpg");
	img_fmt.push_back("jpeg");
	img_fmt.push_back("bmp");

	//Video formats
	vid_fmt.push_back("mp4");
	vid_fmt.push_back("avi");
	vid_fmt.push_back("flv");
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Class destructor
Datasets_Menu::~Datasets_Menu()
{

}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Performs a search along the server DBs directory using the Loop_Options function.
//It shows menus with found valid files (folders, videos, image sequences) and gets
//the user input choices. The search ends when a video or image sequence is selected
//or when there are no more valid files in the current directory.

void Datasets_Menu::Select_Dataset()
{
	bool flg_rdy = false;
	valid = true;

	cout<<"Select Dataset:"<<endl<<endl;
	cout<<"0) Shortcuts"<<endl;
	
	//Perform search through directory until a video or image sequence is selected
	lvl = 0;
	while(!flg_rdy)
	{
		//Read serverDB_path
		strcpy_s(main_path_ch,main_path_str.c_str());
		DIR *dir;
		if((dir = opendir (main_path_ch)) != NULL) 
		{
			//Loop through server DBs
			Loop_Options(dir, main_path_str, fmt);

			if(fmt==1||fmt==2||fmt==0)
				flg_rdy = true;

			lvl++;
		}
	}

	//If it is an image sequence open it
	if(fmt==1)
	{
		strcpy_s(main_path_ch,main_path_str.c_str());
		dir_seq = opendir(main_path_ch);
	}

	//If it is a video open it
	if(fmt==2)
		Source.open(main_path_str);

	//If source not loaded from shortcut, ask if save it in shorcut DB
	if(lvl>1)
		Set_shortcut_DB(fmt, main_path_str);

	if(fmt==0)
		valid = false;

}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Given a directory "*dir" finds all valid files (folders, videos, image sequences)
//and displays a menu if there are various folders or various videos. Then it gets
//the user choice, append the selected path to "main_path" and returns the current 
//selected file category "fmt".

void Datasets_Menu::Loop_Options(DIR *dir, string &main_path, int &fmt)
{
	vector<string> videos, folders;
	string aux, ini_fr;
	int sel, ext, nim=0;

	fmt=0;

	//Loop through folder items
	while ((ent = readdir (dir)) != NULL) 
	{
		//Avoid "." and ".." files
		if(ent->d_namlen>2)
		{
			//Check file extension
			ext = Check_Extension(ent->d_name);

			//Check if image
			if(ext==1)
			{
				nim++;
				
				//if there are more than 5 images in the same driectory consider it as a sequence
				if(nim>5)
				{					
					//main_path.append(ini_fr);
					fmt = 1;
					break;
				}
			}
			
			//Check if video
			ext = Check_Extension(ent->d_name);
			if(ext==2)
			{
				videos.push_back(ent->d_name);
			}

			//Check if folder
			if(ext==3)
			{
				//Save folder names
				folders.push_back(ent->d_name);
			}
		}
	}

	//If there are more than 2 videos show video menu
	if(videos.size()>2)
	{
		//Print menu
		std::cout<<"Select Video:"<<std::endl<<std::endl;
		Display_Menu(videos);

		//Select video
		std::cin>>sel;

		//Store in return variable
		main_path.append(videos[sel-1]);

		//return format
		fmt = 2;
	}
	
	//If there are more than 2 folders show folders menu
	if(folders.size()>1)
	{
		//Print menu
		Display_Menu(folders);

		//Select video
		std::cin>>sel;
		
		//If Shortcut selected read txt and display saved DBs
		if(sel==0&&lvl==0)
		{
			Get_shortcut_DB(fmt, main_path);
		}
		else
		{
			//Store in return variable
			main_path.append(folders[sel-1]);
			main_path.append("\\");

			//return format
			fmt = 3;
		}
	}
	else
	{
		//If there is just one folder auto-open
		if(folders.size()==1&&!(ext==1))
		{
			//Store in return variable
			main_path.append(folders[0]);
			main_path.append("\\");

			//return format
			fmt = 3;
		}
	}
	
	//if not valid format show error
	if(ext==0&&folders.size()<1&&videos.size()<2)
	{
		cout<<"Error: Not valid input files in current directory..."<<endl;
		fmt = 0;
		system("pause");
	}
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Checks the given file specified by "file" and return its category (image, video, 
//folder, other) 

int Datasets_Menu::Check_Extension(string file)
{
	int ext = 0;

	//Loop through image formats
	for(int i=0; i<img_fmt.size(); i++)
	{
		if(file.substr(file.find_last_of(".") + 1) == img_fmt[i])
		{
			ext = 1;
			return ext;
		}
	}

	//Loop through video formats
	for(int i=0; i<vid_fmt.size(); i++)
	{
		if(file.substr(file.find_last_of(".") + 1) == vid_fmt[i])
		{
			ext = 2;
			return ext;
		}
	}

	//Check if folder
	size_t found = file.find(".");
	if(found==string::npos)
	{
		ext = 3;
		return ext;
	}

	return ext;
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Displays the menu specified in "menu"
void Datasets_Menu::Display_Menu(vector<string> menu)
{
	char kch[5];

	for(int i=0; i<menu.size(); i++)
	{
		itoa(i+1,kch,10);
		cout<<kch<<") "<<menu[i]<<endl;
	}
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Gets new frame in "Fr" from an image sequence or video specified by "main_path_ch"

void Datasets_Menu::Get_Frame(cv::Mat &Fr)
{
	//If the selected directory has a sequence or video get frame
	if(valid)
	{
		//Get frame from sequence
		if(fmt==1)
		{
			int len = 0, ext = 0;
			bool read = false;

			while(len<=2||ext!=1)
			{
				ent = readdir (dir_seq);

				//if current file not null, continue
				if(ent!=NULL)
				{
					len = ent->d_namlen;
					ext = Check_Extension(ent->d_name);
					read = true;
				}
				else
				{
					read = false;
					break;
				}
			}
			
			//if current file not null, read frame
			if(read)
			{
				strcpy_s(main_path_ch,main_path_str.c_str());
				strcat_s(main_path_ch,ent->d_name);
				Fr = cv::imread(main_path_ch);
			}
			else
				Fr.release();
		}


		//Get frame from video
		if(fmt==2)
		{
			Source>>Fr;
		}
	}
	else
		Fr.release();
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Set shortcuts to databases if the user chooses to.

void Datasets_Menu::Set_shortcut_DB(int fmt, string path)
{
	char sh;
	cout<<endl<<"Add shortcut to this dataset?(Y/N)"<<endl;
	cin>>sh;
	
	if(sh=='y'||sh=='Y')
	{
		//Assign name to the shortcut
		cout<<endl<<"Assign shortcut name:"<<endl;
		string name;
		cin>>name;

		//Open txt file
		ofstream shtcut;
		shtcut.open("DB_shortcut.txt", ios::app);

		//Conform line to write in txt file
		string line;
		char a[5];
		itoa(fmt,a,10);
		line.append(a);
		line.append(" ");
		line.append(name);
		line.append(">");
		line.append(path);
		line.append("\n");

		//Write line
		shtcut << line;

		shtcut.close();
	}
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
//Get shortcuts to databases if the user chooses option 0 "Shortcut" at level 0

void Datasets_Menu::Get_shortcut_DB(int &fmt, string &path)
{
	//Struct to read DB shortcut txt
	ifstream shtcut("DB_shortcut.txt",ifstream::in);		
	
	if(shtcut.is_open())
	{
		string line;
		vector<string> options;
		vector<int>	fmt_shct;	
		vector<string> names;
		size_t pos;

		//Loop throught DB shortcut txt
		while(getline(shtcut,line))
		{
			//Get format of shortcuts
			fmt_shct.push_back(atoi(&line[0]));

			//Get name of shortcuts
			line.erase(0,2);
			pos = line.find(">");
			names.push_back(line.substr(0,pos));

			//Get path of shortcuts
			line.erase(0,pos+1);
			options.push_back(line);
		}
		shtcut.close();

		//Display options
		cout<<endl<<"Select input source:"<<endl<<endl;
		Display_Menu(names);

		//Select saved DB
		int sel;
		cin>>sel;

		//Get DB information
		fmt = fmt_shct[sel-1];
		path = options[sel-1];
	}
	else
	{
		cout<<"Error: DB_shortcut.txt file not found"<<endl;
		fmt = 0;
		system("pause");
	}
}
//-----------------------------------------------------------------------------------

