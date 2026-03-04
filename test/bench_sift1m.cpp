#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
using GTMatrix = symqg::RowMatrix<int32_t>;

static void load_graph_bin(const char* filename, size_t& n_out, size_t& deg_out,
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
    input.read(reinterpret_cast<char*>(edges.data()),
               static_cast<std::streamsize>(sizeof(uint32_t) * n_out * deg_out));
    if (!input) {
        throw std::runtime_error("Error reading graph data");
    }
}

static float compute_recall(
    const std::vector<std::vector<uint32_t>>& results,
    const GTMatrix& gt,
    uint32_t topk
) {
    size_t nq = results.size();
    size_t total_correct = 0;
    size_t total_num = nq * topk;

    for (size_t i = 0; i < nq; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            uint32_t gt_id = static_cast<uint32_t>(gt(static_cast<long>(i), static_cast<long>(j)));
            for (uint32_t r = 0; r < topk; ++r) {
                if (results[i][r] == gt_id) {
                    ++total_correct;
                    break;
                }
            }
        }
    }
    return static_cast<float>(total_correct) / static_cast<float>(total_num) * 100.0F;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <base.fvecs> <query.fvecs> <groundtruth.ivecs> <graph.bin> <ef_search>\n";
        return 1;
    }

    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];
    std::string graph_file = argv[4];
    size_t ef_search = std::stoul(argv[5]);

    uint32_t topk = 100;
    std::vector<size_t> ef_search_list = {ef_search};

    // --- Load data ---
    std::cout << "=== Loading Data ===" << std::endl;
    DataMatrix base, queries;
    GTMatrix groundtruth;

    symqg::load_vecs<float, DataMatrix>(base_file.c_str(), base);
    std::cout << "Base loaded" << std::endl;
    symqg::load_vecs<float, DataMatrix>(query_file.c_str(), queries);
    std::cout << "Queries loaded" << std::endl;
    symqg::load_vecs<int32_t, GTMatrix>(gt_file.c_str(), groundtruth);
    std::cout << "GT loaded" << std::endl;

    size_t num_base = base.rows();
    size_t dim = base.cols();
    size_t num_queries = queries.rows();

    std::cout << "Base:    " << num_base << " x " << dim << std::endl;
    std::cout << "Queries: " << num_queries << " x " << queries.cols() << std::endl;

    // --- Load external graph ---
    size_t graph_n = 0, graph_deg = 0;
    std::vector<uint32_t> edges;
    load_graph_bin(graph_file.c_str(), graph_n, graph_deg, edges);
    std::cout << "Graph:   " << graph_n << " x " << graph_deg << std::endl;

    // --- Build SymphonyQG index from external graph ---
    std::cout << "\n=== Building QuantizedGraph ===" << std::endl;
    StopW build_timer;

    symqg::QuantizedGraph qg(num_base, graph_deg, dim);
    qg.copy_vectors(base.data());
    std::cout << "Vectors copied" << std::endl;

    // Compute centroid, find nearest real vertex, use as entry point
    int num_threads = omp_get_max_threads();
    std::vector<float> centroid =
        symqg::space::compute_centroid(base.data(), num_base, dim, num_threads);
    symqg::PID entry_point = symqg::space::exact_nn(
        base.data(), centroid.data(), num_base, dim, num_threads, symqg::space::l2_sqr);
    qg.set_ep(entry_point);
    std::cout << "Entry point: " << entry_point << std::endl;

    // Compute quantization codes
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_base; ++i) {
        const uint32_t* neighbors = edges.data() + i * graph_deg;
        std::vector<symqg::Candidate<float>> candidates;
        candidates.reserve(graph_deg);
        for (size_t j = 0; j < graph_deg; ++j) {
            candidates.emplace_back(static_cast<symqg::PID>(neighbors[j]), 0.0F);
        }
        qg.update_qg(static_cast<symqg::PID>(i), candidates);
    }

    float build_secs = build_timer.get_elapsed_mili() / 1000.0F;
    std::cout << "Build time: " << std::fixed << std::setprecision(2)
              << build_secs << " seconds" << std::endl;

    // --- Benchmark queries ---
    std::cout << "\n=== Benchmarking Queries (top-" << topk
              << ", threads=" << num_threads << ") ===" << std::endl;
    std::cout << std::setw(8) << "EF"
              << std::setw(14) << "Recall@" + std::to_string(topk) + "(%)"
              << std::setw(14) << "QPS"
              << std::setw(16) << "Avg Latency(us)"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (size_t ef : ef_search_list) {
        qg.set_ef(ef);

        std::vector<uint32_t> all_results(num_queries * topk);

        StopW query_timer;
        qg.batch_search(queries.data(), static_cast<uint32_t>(num_queries), topk, all_results.data());
        float elapsed_ms = query_timer.get_elapsed_mili();

        // Convert flat results to nested for recall computation
        std::vector<std::vector<uint32_t>> results_nested(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            results_nested[i].assign(
                all_results.data() + i * topk,
                all_results.data() + (i + 1) * topk);
        }

        float recall = compute_recall(results_nested, groundtruth, topk);
        float qps = static_cast<float>(num_queries) / (elapsed_ms / 1000.0F);
        float avg_latency_us = (elapsed_ms * 1000.0F) / static_cast<float>(num_queries);

        std::cout << std::setw(8) << ef
                  << std::setw(14) << std::fixed << std::setprecision(2) << recall
                  << std::setw(14) << std::setprecision(0) << qps
                  << std::setw(16) << std::setprecision(1) << avg_latency_us
                  << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
