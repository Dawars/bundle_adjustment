//
// Created by Komorowicz David on 2020. 07. 22..
//

#include "fmt/os.h"

#include "bundleadjust/MeshWriter.h"


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