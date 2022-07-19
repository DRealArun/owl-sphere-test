// Author: Arun Rajendra Prabhu (arun.prabhu@h-brs.de)
// IVC HBRS

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include "opencv2/imgproc/imgproc_c.h"
#include <Eigen/Dense>

// Obtained from
// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
std::string cvtype2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Eigen::Matrix2Xf generateGridCoordinates(int nH, int nW) {
    // Eigen::RowVectorXf xs = Eigen::RowVectorXf::LinSpaced(nW,0,nW-1);
    // Eigen::RowVectorXf ys = Eigen::RowVectorXf::LinSpaced(nH,0,nH-1);
    Eigen::Matrix2Xf G;
    G.resize(2,nH*nW);
    for (int i = 0; i < nH; ++i) {
        for (int j = 0; j < nW; ++j) {
            G(0, (i*nW)+j) = i;
            G(1, (i*nW)+j) = j;
        }
    }
    return G;
}

Eigen::Matrix4Xf projectPix2Camera(cv::Mat depth, Eigen::Matrix4f K, int zNear, int zFar, Eigen::MatrixXi &Mask) {
    Eigen::Matrix4Xf camCoords;
    camCoords.resize(4, depth.rows*depth.cols);
    Mask.resize(1, depth.rows*depth.cols);

    cv::Mat img1dGray = cv::Mat(depth.rows, depth.cols, depth.type(), cvScalar(0.));
    // std::cout << cvtype2str(depth.type()) << " type" << std::endl;
    for(int row = 0; row < depth.rows; ++row) {
        for(int col = 0; col < depth.cols; ++col) {
            float& pix = depth.ptr<float>(row)[col];
            float modpix;
            if ((pix >= zNear) && (pix <= zFar)) {
                modpix = pix;
                Mask(0, (row*depth.cols)+col) = 1;
            }
            if (pix > zFar) {
                modpix = zFar;
                Mask(0, (row*depth.cols)+col) = 0;
            }
            if (pix < zNear) {
                modpix = zFar;
                Mask(0, (row*depth.cols)+col) = 0;
            }
            pix = modpix;
            float normCol = (col  * 2.0 / depth.cols) - 1;
            float normRow = (row  * 2.0 / depth.rows) - 1;
            camCoords(0, (row*depth.cols)+col) = modpix*normCol;
            camCoords(1, (row*depth.cols)+col) = -1*modpix*normRow;
            camCoords(2, (row*depth.cols)+col) = -1*modpix;
            camCoords(3, (row*depth.cols)+col) = 1;
        }
    }
    Eigen::Matrix4f Kinv = K.inverse();
    return Kinv*camCoords;
}

Eigen::Matrix4Xf projectPix2Camera2(cv::Mat depth, Eigen::Matrix4f K, float zNear, float zFar) {
    // function for debugging only
    Eigen::Matrix4Xf camCoords;
    camCoords.resize(4, 8);
    std::vector<float> depths= {zNear, zFar};
    std::vector<int> rows= {0, depth.rows};
    std::vector<int> cols= {0, depth.cols};
    int count = 0;
    for (float pix : depths) {
        for(int row : rows) {
            for(int col : cols) {
                float normCol = (col  * 2.0 / depth.cols) - 1;
                float normRow = (row  * 2.0 / depth.rows) - 1;
                camCoords(0, count) = pix*normCol;
                camCoords(1, count) = -1*pix*normRow;
                camCoords(2, count) = -1*pix;
                camCoords(3, count) = 1;
                count += 1;
            }
        }
    }
    std::cout << "cam coords: \n" << camCoords.transpose() << std::endl;
    Eigen::Matrix4f Kinv = K.inverse();
    std::cout << "inverse of K " <<  Kinv << std::endl;
    return Kinv*camCoords;
}

Eigen::Matrix4Xf projectCam2World(Eigen::Matrix4Xf camCoords, Eigen::Matrix4f P) {
    return P*camCoords;
}

Eigen::Matrix4Xf projectWorld2Cam(Eigen::Matrix4Xf wrldCoords, Eigen::Matrix4f P) {
    Eigen::Matrix4f Pinv = P.inverse();
    return Pinv*wrldCoords;
}

Eigen::Matrix2Xf projectCam2Pix(Eigen::Matrix4f camCoords, Eigen::Matrix4f K) {
    Eigen::Matrix3Xf pts = (K*camCoords).topRows(3);
    Eigen::Matrix3Xf temp = pts.array().rowwise() / pts.row(2).array();
    return temp.topRows(2);
    // While using the output of this function kindly transpose
}