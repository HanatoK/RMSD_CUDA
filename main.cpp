#include "rmsd_cuda.h"

#include <fstream>
// #include <cstdlib>
#include <fmt/format.h>
#include <iostream>
#include <random>

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
host_vector<host_vector<AtomPosition>> readFromFile(const std::string& filename) {
    std::ifstream ifs_traj(filename.c_str());
    std::string line;
    std::vector<std::string> fields;
    host_vector<host_vector<AtomPosition>> atom_sets;
    while (std::getline(ifs_traj, line)) {
        splitString(line, std::string{" "}, fields);
        if (fields.size() > 0 && (fields.size() % 3 == 0)) {
            host_vector<AtomPosition> tmp_set;
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

host_vector<host_vector<AtomPosition>> randomPositions(const size_t num_atoms = 100000, const size_t num_frames = 100) {
    host_vector<host_vector<AtomPosition>> atom_sets;
    host_vector<AtomPosition> ref_atoms;
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    for (size_t j = 0; j < num_atoms; ++j) {
        ref_atoms.push_back({dis(gen), dis(gen), dis(gen)});
    }
    atom_sets.push_back(ref_atoms);
    std::normal_distribution<> dis_norm(0.0, 0.5);
    for (size_t i = 1; i < num_frames; ++i) {
        host_vector<AtomPosition> pos = ref_atoms;
        for (size_t j = 0; j < num_atoms; ++j) {
            pos[j].x = atom_sets[i-1][j].x + dis_norm(gen);
            pos[j].y = atom_sets[i-1][j].y + dis_norm(gen);
            pos[j].z = atom_sets[i-1][j].z + dis_norm(gen);
        }
        atom_sets.push_back(pos);
    }
    return atom_sets;
}

int main(int argc, char* argv[]) {
    // read data from file
    const host_vector<host_vector<AtomPosition>> atom_positions = argc < 2 ? randomPositions() : readFromFile(argv[1]);
    // if (argc < 2) return 1;
#if defined (USE_NR)
    std::cout << "Use NR algorithm to compute the eigenvectors and eigenvalues\n";
#endif
#if defined (USE_CUDA_GRAPH)
    std::cout << "Use CUDA Graphs\n";
#endif
    nvtxEventAttributes_t nvtx_event;
    nvtx_event.version = NVTX_VERSION;
    nvtx_event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_event.colorType = NVTX_COLOR_ARGB;
    nvtx_event.color = 0xda7acd;
    nvtx_event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_event.message.ascii = "RMSD";
    // initialize the rotation CV
    OptimalRotation rot(atom_positions[0].size());
    // indirect compute rmsd: rotate and use RMSD equation directly
#if !defined (USE_CUDA_GRAPH)
    for (size_t i = 0; i < atom_positions.size() - 1; ++i) {
        // nvtxRangePushEx(&nvtx_event);
        // update atomic coordinates
        rot.updateAtoms(atom_positions[i+1]);
        // update reference frames
        rot.updateReference(atom_positions[i]);
        // compute the optimal rotation matrix
        rot.calculateOptimalRotationMatrix();
        // nvtxRangePop();
        // compute the optimal rmsd
        std::cout << fmt::format("Optimal RMSD between frame {:4d} and {:4d} is {:15.12f}\n", i, i+1, rot.minimalRMSD(atom_positions[i+1]));
    }
#endif
    // direct compute: using the equation in page 1855, "Using Quaternions to Calculate RMSD"
//     std::printf("Using the equation in page 1855:\n");
    for (size_t i = 0; i < atom_positions.size() - 1; ++i) {
        if (i > 0) nvtxRangePushEx(&nvtx_event);
        // update atomic coordinates
        rot.updateAtoms(atom_positions[i+1]);
        // update reference frames
        rot.updateReference(atom_positions[i]);
        // compute the optimal rotation matrix
        rot.calculateOptimalRotationMatrix();
        // compute the optimal rmsd
        std::cout << fmt::format("Optimal RMSD between frame {:4d} and {:4d} is {:15.12f}\n", i, i+1, rot.minimalRMSD());
        if (i > 0) nvtxRangePop();
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
