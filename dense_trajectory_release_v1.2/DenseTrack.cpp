#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace cv;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

namespace {
	class ValidTrack
	{
	public:
		std::vector<Mat>::size_type frame_num;
		float mean_x;
		float mean_y;
		float var_x;
		float var_y;
		float length;
		float scale;
		float x_pos;
		float y_pos;
		float t_pos;
		std::vector<Point2f> coords;
		std::vector<Point2f> traj;
		std::vector<float> hog;
		std::vector<float> hof;
		std::vector<float> mbhX;
		std::vector<float> mbhY;

		ValidTrack() :
			frame_num(0), mean_x(0), mean_y(0), var_x(0), var_y(0),
			length(0), scale(0), x_pos(0), y_pos(0), t_pos(0) {
		}

		ValidTrack(std::vector<Mat>::size_type frame_num_,
			float mean_x_, float mean_y_, float var_x_, float var_y_,
			float length_, float scale_, float x_pos_, float y_pos_, float t_pos_,
			const std::vector<Point2f>& coords_, const std::vector<Point2f>& traj_,
			const std::vector<float>& hog_, const std::vector<float>& hof_,
			const std::vector<float>& mbhX_, const std::vector<float>& mbhY_) :
			frame_num(frame_num_), mean_x(mean_x_), mean_y(mean_y_),
			var_x(var_x_), var_y(var_y_), length(length_), scale(scale_),
			x_pos(x_pos_), y_pos(y_pos_), t_pos(t_pos_), coords(coords_), traj(traj_),
			hog(hog_), hof(hof_), mbhX(mbhX_), mbhY(mbhY_) {
		}

		PyObject* toPython() {
			return Py_BuildValue("(ifffffffffNNNNNN)",
					frame_num, mean_x, mean_y, var_x, var_y, length,
					scale, x_pos, y_pos, t_pos,
					toPython(coords), toPython(traj), toPython(hog),
					toPython(hof), toPython(mbhX), toPython(mbhY));
		}

	private:
		PyObject* toPython(const std::vector<float>& values) {
			// This could be sped up more by creating a Numpy array here.
			PyObject* py_list = PyList_New(values.size());
			for (size_t i = 0; i < values.size(); i++)
				PyList_SetItem(py_list, i, Py_BuildValue("f", values[i]));
			return py_list;
		}

		PyObject* toPython(const std::vector<Point2f>& values) {
			PyObject* py_list = PyList_New(values.size());
			for (size_t i = 0; i < values.size(); i++)
				PyList_SetItem(py_list, i, Py_BuildValue("[ff]", values[i].x, values[i].y));
			return py_list;
		}
	};

	class GILState {
		PyGILState_STATE gstate;
	public:
		GILState() {
			// See http://stackoverflow.com/questions/35774011/segment-fault-when-creating-pylist-new-in-python-c-extention
			// for why we need line of code below
			// Note that this is only needed for ctypes.CDLL and not for ctypes.PyDLL but does no harm for the latter.
			gstate = PyGILState_Ensure();
		}
		~GILState() {
			PyGILState_Release(gstate);
		}
	};

	int createDirectory(const char* path) {
		const char* end = strrchr(path, '/');
		if (end == NULL || end == path)
			return 0;
		size_t len = end - path;
		char dir_path[300];
		if (len + 1 > sizeof(dir_path)) {
			errno = ENAMETOOLONG;
			return -1;
		}
		memcpy(dir_path, path, len);
		dir_path[len] = '\0';
		for (char* p = dir_path + 1; *p; p++) {
			if (*p == '/') {
				*p = '\0';
				if (mkdir(dir_path, S_IRWXU) < 0 && errno != EEXIST)
					return -1;
				*p = '/';
			}
		}
		if (mkdir(dir_path, S_IRWXU) < 0 && errno != EEXIST)
			return -1;
		return 0;
	}
}

extern "C"
//PyObject* densetrack(char* video, int _start_frame, int _end_frame, int _track_length, int _min_distance,
PyObject* densetrack(unsigned char *frames, size_t len, size_t rows, size_t cols, int _track_length, 
                     int _min_distance, int _patch_size, int _nxy_cell, int _nt_cell, 
                     int _scale_num, int _init_gap, 
                     int _poly_n, double _poly_sigma, const char* image_pattern)
{
//	fprintf(stderr, "C++ - poly_sigma: %f\n", _poly_sigma);

	GILState gstate;
	if (PyArray_API == NULL) {
		import_array();
	}

	std::vector<ValidTrack> valid_tracks;

	// Note that this opens a block closed by Py_END_ALLOW_THREADS.
	// https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code
	// Variables may need to be declared outside that block.
	Py_BEGIN_ALLOW_THREADS

	// create a vector of cv::mat to hold frames of video
	std::vector<Mat> video;
	video.reserve(len);
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
	bool first_image = true;

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
			if (show_track == 1 || image_pattern != NULL)
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
			if (show_track == 1 || image_pattern != NULL)
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
		if (show_track == 1 || image_pattern != NULL)
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
				if((show_track == 1 || image_pattern != NULL) && iScale == 0)
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
						// for spatio-temporal pyramid
						float x_pos = std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999);
						float y_pos = std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999);
						float t_pos = std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999);
						std::vector<float> hog;
						std::vector<float> hof;
						std::vector<float> mbhX;
						std::vector<float> mbhY;
						PrintDesc(iTrack->hog, hogInfo, trackInfo, hog);
						PrintDesc(iTrack->hof, hofInfo, trackInfo, hof);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, mbhX);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, mbhY);
						valid_tracks.push_back(ValidTrack(frame_num, mean_x, mean_y,
										var_x, var_y, length, fscales[iScale],
										x_pos, y_pos, t_pos, trajectory_copy, trajectory,
										hog, hof, mbhX, mbhY));
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
		if (image_pattern != NULL) {
			char path[300];
			snprintf(path, sizeof(path), image_pattern, frame_num);
			if (first_image) {
				first_image = false;
				if (createDirectory(path) < 0) {
					Py_BLOCK_THREADS
					return PyErr_SetFromErrno(PyExc_OSError);
				}
			}
			try {
				cv::imwrite(path, image);
			}
			catch (const cv::Exception& ex) {
				Py_BLOCK_THREADS
				PyErr_SetString(PyExc_RuntimeError, ex.what());
				Py_RETURN_NONE;
			}
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrack");

	std::vector<std::list<Track> >().swap(xyScaleTracks);
	std::vector<Mat>().swap(video);
	image.release();
	prev_grey.release();
	grey.release();
	std::vector<Mat>().swap(prev_grey_pyr);
	std::vector<Mat>().swap(grey_pyr);
	std::vector<Mat>().swap(flow_pyr);
	std::vector<Mat>().swap(prev_poly_pyr);
	std::vector<Mat>().swap(poly_pyr);

	Py_END_ALLOW_THREADS

	int cell_size = nxy_cell * nxy_cell * nt_cell;
	// PyList_Append increases the ref count (unlike PyList_SetItem)
	// https://stackoverflow.com/questions/3512414/does-this-pylist-appendlist-py-buildvalue-leak
	PyObject* dtype = PyList_New(16);
	int idx = 0;
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "frame_num", "i", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mean_x", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mean_y", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "var_x", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "var_y", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "length", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "scale", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "x_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "y_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "t_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, (i, i))", "coords", "f", track_length + 1, 2));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, (i, i))", "trajectory", "f", track_length, 2));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "hog", "f", 8 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "hof", "f", 9 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mbh_x", "f", 8 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mbh_y", "f", 8 * cell_size));
	PyArray_Descr* descr;
	PyArray_DescrConverter(dtype, &descr);
	Py_DECREF(dtype);
	npy_intp dims[1];
	dims[0] = valid_tracks.size();
	PyArrayObject* py_tracks =
		(PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL, NULL, 0, NULL);
	if (!py_tracks) {
		fprintf(stderr, "Error creating Numpy array\n");
		Py_RETURN_NONE;
	}

	npy_intp stride = PyArray_STRIDES(py_tracks)[0];
	char* bytes = PyArray_BYTES(py_tracks);
	for (size_t i = 0; i < valid_tracks.size(); i++) {
		PyObject* item = valid_tracks[i].toPython();
		PyArray_SETITEM(py_tracks, bytes + (stride * i), item);
		Py_DECREF(item);
		// Clear the track data
		ValidTrack tmp;
		std::swap(tmp, valid_tracks[i]);
	}

	return (PyObject*) py_tracks;
}
