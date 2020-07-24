//
// Created by Komorowicz David on 2020. 07. 22..
//

#include "fmt/os.h"

#include "bundleadjust/MeshWriter.h"
#include "bundleadjust/BundleAdjustment.h"

void MeshWriter::WriteToPLYFile(std::string filename, std::vector<Eigen::Vector3f> points,  std::vector<Eigen::Vector3i> colors, std::vector<Eigen::Vector3f> cams) {
    auto of = fmt::output_file(filename);

    of.print("ply"
             "\nformat ascii 1.0"
             "\nelement vertex {}"
             "\nproperty float x"
             "\nproperty float y"
             "\nproperty float z"
             "\nproperty uchar red"
             "\nproperty uchar green"
             "\nproperty uchar blue"
             "\nend_header\n", points.size() + cams.size());

    for (int i = 0; i < cams.size(); ++i)  {
        auto cam = cams[i];
        of.print("{0:.4f} {1:.4f} {2:.4f} 0 255 0\n", cam[0], cam[1], cam[2]);
    }

    for (int i = 0; i < points.size(); ++i) {
        auto point = points[i];
        Eigen::Vector3i color = colors[i];
        of.print("{0:.4f} {1:.4f} {2:.4f} {3} {4} {5}\n", point[0], point[1], point[2], color(0), color(1), color(2));
    }
    of.close();
}


void MeshWriter::WritePLYForInits(BundleAdjustment & ba, std::string output_dir) {
    const int total_frames = ba.dataset->getNumFrames() - 1;
    for(int other_frame_index=1; other_frame_index<total_frames; other_frame_index++) {


        auto of = fmt::output_file(output_dir + "/0_to_" + std::to_string(other_frame_index) + ".ply");

        int total_points_in_mesh = 0;
        std::vector<Eigen::Vector3f> points;  
        std::vector<Eigen::Vector3i> colors;

        for(int i=0; i<ba.dataset->getNumObservations(); i++) {
            const int obs_cam_index = ba.dataset->getObsCam(i);

            if(obs_cam_index == 0) {

                
                const int point_index = ba.dataset->getObsPoint(i);

                if(point_index == -1) continue;

                for(int j=0; j<ba.dataset->getNumObservations(); j++) {
                    if(point_index == ba.dataset->getObsPoint(j) && ba.dataset->getObsCam(j) == other_frame_index) {
                        total_points_in_mesh += 1;
                        Eigen::Vector3f p;
                        Eigen::Vector3i c;

                        auto X = ba.getPoint(point_index);
                        p << X[0], X[1], X[2];

                        c = ba.dataset->getPointColor(i);
                        points.push_back(p);
                        colors.push_back(c);
                        break;
                    }
                }
            }
        }

        of.print("ply"
                "\nformat ascii 1.0"
                "\nelement vertex {}"
                "\nproperty float x"
                "\nproperty float y"
                "\nproperty float z"
                "\nproperty uchar red"
                "\nproperty uchar green"
                "\nproperty uchar blue"
                "\nend_header\n", total_points_in_mesh);


        auto cam = ba.getTranslation(0);
        auto other_cam = ba.getTranslation(other_frame_index);
        of.print("{0:.4f} {1:.4f} {2:.4f} 0 255 0\n", cam[0], cam[1], cam[2]);
        of.print("{0:.4f} {1:.4f} {2:.4f} 0 255 0\n", other_cam[other_frame_index], other_cam[other_frame_index], other_cam[other_frame_index]);

        for (int i = 0; i < points.size(); ++i) {
            auto point = points[i];
            Eigen::Vector3i color = colors[i];
            of.print("{0:.4f} {1:.4f} {2:.4f} {3} {4} {5}\n", point[0], point[1], point[2], color(0), color(1), color(2));
        }
        of.close();
    }
}