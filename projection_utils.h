#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

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

// xs = np.linspace(0, cols - 1, cols) / float(cols - 1) * 2 - 1
// ys = np.linspace(0, rows - 1, rows) / float(rows - 1) * 2 - 1

// M at is already flattened.
Eigen::Matrix4Xf projectPix2Camera(cv::Mat depth, Eigen::Matrix4f K, int zNear, int zFar) { 
    // Get valid depth values and its associated indices.
    Eigen::Matrix4Xf camCoords;
    camCoords.resize(4, depth.rows*depth.cols);
    cv::Mat img1dGray = cv::Mat(depth.rows, depth.cols, depth.type(), cvScalar(0.));
    std::cout << cvtype2str(depth.type()) << " type" << std::endl;
    double minVal; 
    double maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    std::cout << "Max:Min " << maxVal << ":" << minVal << std::endl;
    for(int row = 0; row < depth.rows; ++row) {
        for(int col = 0; col < depth.cols; ++col) {
            float& pix = depth.ptr<float>(row)[col];
            float modpix;
            if ((pix >= zNear) && (pix <= zFar)) {
                modpix = pix;
            }
            if (pix > zFar) {
                modpix = zFar;
            }
            if (pix < zNear) {
                modpix = zNear;
            }
            // float normPix = (modpix  * 2 / zFar) - 1;
            // float normCol = (col  * 2 / depth.cols) - 1;
            // float normRow = (row  * 2 / depth.rows) - 1;
            camCoords(0, (row*depth.cols)+col) = modpix*col;
            camCoords(1, (row*depth.cols)+col) = -1*modpix*row;
            camCoords(2, (row*depth.cols)+col) = -1*modpix;
            camCoords(3, (row*depth.cols)+col) = 1;
        }
    }
    // for (auto id: validIds) {
    //      img1dGray.ptr<float>(id[0])[id[1]] = 255;
    // }
    // img1dGray.convertTo(img1dGray, CV_8UC1);float, 4> 
    // cv::imshow("Depth 2", img1dGray);
    // std::cout << camCoords << std::endl;
    Eigen::Matrix4f Kinv = K.inverse();
    return Kinv*camCoords;
}

Eigen::Matrix4f projectCam2World(Eigen::Matrix4f camCoords, Eigen::Matrix4f P) {
    return P*camCoords;
}

Eigen::Matrix4f projectWorld2Cam(Eigen::Matrix4f wrldCoords, Eigen::Matrix4f P) {
    Eigen::Matrix4f Pinv = P.inverse();
    return Pinv*wrldCoords;
}

Eigen::Matrix2Xf projectCam2Pix(Eigen::Matrix4f camCoords, Eigen::Matrix4f K) {
    Eigen::Matrix3Xf pts = (K*camCoords).topRows(3);
    Eigen::Matrix3Xf temp = pts.array().rowwise() / pts.row(2).array();
    return temp.topRows(2);
    // While using the output of this function kindly transpose
}

// def get_warped_image(src_img, points, mask):
//     rows, cols, _ = src_img.shape
//     warped_img = np.ones_like(src_img)*255
//     xs = np.linspace(0, cols - 1, cols).astype(np.int32)
//     ys = np.linspace(0, rows - 1, rows).astype(np.int32)
//     c, r = np.meshgrid(xs, ys, sparse=False)
//     r = r.flatten()[mask]
//     c = c.flatten()[mask]
//     for count, (i, j) in enumerate(zip(r, c)):
//         pt = [points[count, 0]*cols, points[count, 1]*rows]
//         if pt[0] >= cols or pt[0] < 0 or pt[1] >= rows or pt[1] < 0:
//             continue
//         else:
//             warped_img[int(pt[1]), cols-1-int(pt[0]), :] = src_img[i,j,:]
//     return warped_img

// cv::Mat warpImage(cv::Mat img, Eigen::Matrix2Xf pts) {

// } 