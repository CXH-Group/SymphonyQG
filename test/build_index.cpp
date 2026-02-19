#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../symqglib/common.hpp"
#include "../symqglib/qg/qg.hpp"
#include "../symqglib/space/l2.hpp"
#include "../symqglib/space/space.hpp"
#include "../symqglib/utils/io.hpp"
#include "../symqglib/utils/stopw.hpp"

using DataMatrix = symqg::RowMatrix<float>;

// Load graph in bin format: int32 num_points, int32 degree, then uint32[num*degree]
void load_graph_bin(const char* filename, size_t& n_out, size_t& deg_out,
                    std::vector<uint32_t>& edges) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("Cannot open graph file: ") + filename);
    }
    int32_t num = 0, deg = 0;
    input.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&deg), sizeof(int32_t));
    n_out = static_cast<size_t>(num);
    deg_out = static_cast<size_t>(deg);
    edges.resize(n_out * deg_out);
    input.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * n_out * deg_out);
    if (!input) {
        throw std::runtime_error("Error reading graph data");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <base.fvecs> <graph.bin> <output_index_file>\n";
        return 1;
    }

    std::string base_file = argv[1];
    std::string graph_file = argv[2];
    std::string index_file = argv[3];

    // Load base vectors
    DataMatrix base;
    symqg::load_vecs<float, DataMatrix>(base_file.c_str(), base);
    size_t num_base = base.rows();
    size_t dim = base.cols();
    std::cout << "Base: " << num_base << " x " << dim << '\n';

    // Load external graph
    size_t graph_n = 0, graph_deg = 0;
    std::vector<uint32_t> edges;
    load_graph_bin(graph_file.c_str(), graph_n, graph_deg, edges);
    std::cout << "Graph: " << graph_n << " x " << graph_deg << '\n';

    if (graph_n != num_base) {
        std::cerr << "Error: graph num_points (" << graph_n
                  << ") != base num_points (" << num_base << ")\n";
        return 1;
    }

    size_t degree = graph_deg;

    // Create QuantizedGraph and copy vectors
    symqg::QuantizedGraph qg(num_base, degree, dim);
    qg.copy_vectors(base.data());

    // Compute entry point (closest to centroid)
    int num_threads = omp_get_max_threads();
    std::vector<float> centroid =
        symqg::space::compute_centroid(base.data(), num_base, dim, num_threads);
    symqg::PID entry_point = symqg::space::exact_nn(
        base.data(), centroid.data(), num_base, dim, num_threads, symqg::space::l2_sqr);
    qg.set_ep(entry_point);
    std::cout << "Entry point: " << entry_point << '\n';

    // Populate quantization data from external graph
    std::cout << "Computing quantization codes...\n";
    StopW timer;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_base; ++i) {
        const uint32_t* neighbors = edges.data() + i * degree;
        std::vector<symqg::Candidate<float>> candidates;
        for (size_t j = 0; j < degree; ++j) {
            candidates.emplace_back(static_cast<symqg::PID>(neighbors[j]), 0.0F);
        }
        qg.update_qg(static_cast<symqg::PID>(i), candidates);
    }

    float secs = timer.get_elapsed_mili() / 1000.0F;
    std::cout << "Quantization time: " << secs << " seconds\n";

    // Save
    auto parent = std::filesystem::path(index_file).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    qg.save_index(index_file.c_str());
    std::cout << "Saved to: " << index_file << '\n';

    return 0;
}
