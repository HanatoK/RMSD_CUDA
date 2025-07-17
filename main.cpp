#include "rmsd_cuda.h"

#include <fstream>
// #include <cstdlib>
#include <fmt/format.h>
#include <iostream>

// split a string by a specified separator
void splitString(const std::string& data, const std::string& delim, std::vector<std::string>& dest) {
    if (delim.empty()) {
        dest.push_back(data);
        return;
    }
    size_t index = 0, new_index = 0;
    std::string tmpstr;
    while (index != data.length()) {
        new_index = data.find(delim, index);
        if (new_index != std::string::npos) tmpstr = data.substr(index, new_index - index);
        else tmpstr = data.substr(index, data.length());
        if (!tmpstr.empty()) {
            dest.push_back(tmpstr);
        }
        if (new_index == std::string::npos) break;
        index = new_index + 1;
    }
}

// read data from file
std::vector<std::vector<AtomPosition>> readFromFile(const std::string& filename) {
    std::ifstream ifs_traj(filename.c_str());
    std::string line;
    std::vector<std::string> fields;
    std::vector<std::vector<AtomPosition>> atom_sets;
    while (std::getline(ifs_traj, line)) {
        splitString(line, std::string{" "}, fields);
        if (fields.size() > 0 && (fields.size() % 3 == 0)) {
            std::vector<AtomPosition> tmp_set;
            for (size_t i = 0; i < fields.size() / 3; ++i) {
                AtomPosition atom_pos;
                atom_pos.x = std::stod(fields[i*3]);
                atom_pos.y = std::stod(fields[i*3+1]);
                atom_pos.z = std::stod(fields[i*3+2]);
                tmp_set.push_back(atom_pos);
            }
            atom_sets.push_back(tmp_set);
        }
        fields.clear();
    }
    return atom_sets;
}

int main(int argc, char* argv[]) {
    // read data from file
    if (argc < 2) return 1;
    const std::vector<std::vector<AtomPosition>> atom_positions = readFromFile(argv[1]);
    // initialize the rotation CV
    OptimalRotation rot(atom_positions[0].size());
    // indirect compute rmsd: rotate and use RMSD equation directly
    for (size_t i = 0; i < atom_positions.size() - 1; ++i) {
        // update atomic coordinates
        rot.updateAtoms(atom_positions[i+1]);
        // update reference frames
        rot.updateReference(atom_positions[i]);
        // compute the optimal rotation matrix
        rot.calculateOptimalRotationMatrix();
        // compute the optimal rmsd
        std::cout << fmt::format("Optimal RMSD between frame {:4d} and {:4d} is {:15.12f}\n", i, i+1, rot.minimalRMSD(atom_positions[i+1]));
    }
    // direct compute: using the equation in page 1855, "Using Quaternions to Calculate RMSD"
//     std::printf("Using the equation in page 1855:\n");
    for (size_t i = 0; i < atom_positions.size() - 1; ++i) {
        // update atomic coordinates
        rot.updateAtoms(atom_positions[i+1]);
        // update reference frames
        rot.updateReference(atom_positions[i]);
        // compute the optimal rotation matrix
        rot.calculateOptimalRotationMatrix();
        // compute the optimal rmsd
        std::cout << fmt::format("Optimal RMSD between frame {:4d} and {:4d} is {:15.12f}\n", i, i+1, rot.minimalRMSD());
    }
    // optimal RMSD between any two frames
    std::cout << "Optimal RMSD between any two frames:\n";
    for (size_t i = 0; i < atom_positions.size(); ++i) {
        // update reference frames
        rot.updateReference(atom_positions[i]);
        for (size_t j = i; j < atom_positions.size(); ++j) {
            // update atomic coordinates
            rot.updateAtoms(atom_positions[j]);
            // compute the optimal rotation matrix
            rot.calculateOptimalRotationMatrix();
            std::cout << fmt::format("Optimal RMSD between frame {:4d} and {:4d} is {:15.12f}\n", i, j, rot.minimalRMSD());
        }
    }
    return 0;
}
