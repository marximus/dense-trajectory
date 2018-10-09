#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

using namespace cv;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

extern "C"
//PyObject* densetrack(char* video, int _start_frame, int _end_frame, int _track_length, int _min_distance,
PyObject* densetrack(unsigned char *frames, size_t len, size_t rows, size_t cols, int _track_length, 
                     int _min_distance, int _patch_size, int _nxy_cell, int _nt_cell, 
                     int _scale_num, int _init_gap, 
                     int _poly_n, double _poly_sigma)
{
//	fprintf(stderr, "C++ - poly_sigma: %f\n", _poly_sigma);

	// See http://stackoverflow.com/questions/35774011/segment-fault-when-creating-pylist-new-in-python-c-extention
	// for why we need line of code below
	PyGILState_STATE gstate = PyGILState_Ensure();

	// create a vector of cv::mat to hold frames of video
	std::vector<Mat> video;
	for (size_t k = 0; k < len; k++) {
		Mat frame = Mat(rows, cols, CV_8UC1, (frames + k*rows*cols));
		video.push_back(frame);
	}

	track_length = _track_length;
	min_distance = _min_distance;
	patch_size = _patch_size;
	nxy_cell = _nxy_cell;
	nt_cell = _nt_cell;
	scale_num = _scale_num;
	init_gap = _init_gap;

	PyObject *py_tracks = PyList_New(0);
	if (!py_tracks) {
		fprintf(stderr, "Error creating python list\n");
		Py_RETURN_NONE;
	}

//	VideoCapture capture;
//	char* video = argv[1];
//	int flag = arg_parse(argc, argv);  // in arg_parse, `flag` is set if start_frame or end_frame is passed
	int flag = false;
//	capture.open(video);

//	if(!capture.isOpened()) {
//		fprintf(stderr, "Could not initialize capturing..\n");
//		Py_RETURN_NONE;
//	}

//	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

//	if(flag)
//		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrack", 0);

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0); // for optical flow

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points

//	while(true) {
	for(std::vector<Mat>::size_type frame_num = 0; frame_num != video.size(); frame_num++) {
		Mat frame;
		int i, j, c;

		// get a new frame
//		capture >> frame;
		frame = video[frame_num];
		if(frame.empty())
			break;

//		if(frame_num < start_frame || frame_num > end_frame) {
//			frame_num++;
//			continue;
//		}

//		if(frame_num == start_frame) {
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			// original code would have had an RGB image here but the function only uses the shape
			// information of the frame so it is alright that it is grayscale
			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);

			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);

			xyScaleTracks.resize(scale_num);

//			frame.copyTo(image);
//			cvtColor(image, prev_grey, CV_BGR2GRAY);
			frame.copyTo(prev_grey);
			cvtColor(frame, image, CV_GRAY2BGR);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			// my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, _poly_n, _poly_sigma);

//			frame_num++;
			continue;
		}

		init_counter++;
//		frame.copyTo(image);
//		cvtColor(image, grey, CV_BGR2GRAY);
		frame.copyTo(grey);
		cvtColor(frame, image, CV_GRAY2BGR);

		// compute optical flow for all scales once
		//my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, _poly_n, _poly_sigma);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i] * fscales[iScale];

					// Create a copy of the track coordinates because they are normalized by IsValid() call below.
					std::vector<Point2f> trajectory_copy(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory_copy[i] = iTrack->point[i] * fscales[iScale];

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length)) {
//						PyObject *py_track_info = PyList_New(0);
						PyObject *py_track_coords = PyList_New(0);
						PyObject *py_track_traj = PyList_New(0);
						PyObject *py_track_hog = PyList_New(0);
						PyObject *py_track_hof = PyList_New(0);
						PyObject *py_track_mbhx = PyList_New(0);
						PyObject *py_track_mbhy = PyList_New(0);

						// output the original trajectory coordinates mapped to original resolution
						for (int i = 0; i != trajectory_copy.size(); i++)
							PyList_Append(py_track_coords, Py_BuildValue("[ff]", trajectory_copy[i].x, trajectory_copy[i].y));

						// for spatio-temporal pyramid
						float x_pos = std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999);
						float y_pos = std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999);
						float t_pos = std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999);
//						py_track_info = Py_BuildValue("ifffffffff", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale], x_pos, y_pos, t_pos);

						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							PyList_Append(py_track_traj, Py_BuildValue("[ff]", trajectory[i].x, trajectory[i].y));

						PrintDesc(iTrack->hog, hogInfo, trackInfo, py_track_hog);
						PrintDesc(iTrack->hof, hofInfo, trackInfo, py_track_hof);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, py_track_mbhx);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, py_track_mbhy);

						// Using "N" does NOT increment the reference count. Using "O" WILL increment the reference count
						// TODO: Need to check if the reference counting below is correct  "(NNNNNNN)"
//						PyObject *py_track = Py_BuildValue("(NNNNNNN)", py_track_info, py_track_coords,
						PyObject *py_track = Py_BuildValue("(ifffffffffNNNNNN)",
										frame_num, mean_x, mean_y, var_x, var_y, length,
										fscales[iScale], x_pos, y_pos, t_pos,
										py_track_coords, py_track_traj, py_track_hog,
										py_track_hof, py_track_mbhx, py_track_mbhy);
						PyList_Append(py_tracks, py_track);
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every initGap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

//		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrack", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrack");

	PyGILState_Release(gstate);

	return py_tracks;
}
