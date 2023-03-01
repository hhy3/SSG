//
// Created by 付聪 on 2017/6/21.
//

#include <chrono>

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"


template <typename T>
T* load_data(const char* filename, unsigned& num, unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Open file error" << std::endl;
    exit(-1);
  }

  in.read((char*)&dim, 4);

  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  T* data = new T[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * sizeof(float));
  }
  in.close();

  return data;
}
int main(int argc, char** argv) {
  if (argc < 7) {
    std::cout << "./run data_file query_file ssg_path L K gt_path [seed]"
              << std::endl;
    exit(-1);
  }

  if (argc == 8) {
    unsigned seed = (unsigned)atoi(argv[7]);
    srand(seed);
    std::cerr << "Using Seed " << seed << std::endl;
  }

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim;
  float* data_load = nullptr;
  data_load = ::load_data<float>(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim;
  float* query_load = nullptr;
  query_load = ::load_data<float>(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  std::cerr << "GT Path: " << argv[6] << std::endl;
  unsigned gt_k;
  int* gt = ::load_data<int>(argv[6], query_num, gt_k);

  assert(dim == query_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexSSG index(dim, points_num, efanna2e::FAST_L2,
                           (efanna2e::Index*)(&init_index));

  std::cerr << "SSG Path: " << argv[3] << std::endl;
  std::cerr << "Result Path: " << argv[6] << std::endl;

  index.Load(argv[3]);
  index.OptimizeGraph(data_load);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  // Warm up
  for (int loop = 0; loop < 3; ++loop) {
    for (unsigned i = 0; i < 10; ++i) {
      index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
    }
  }

  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  printf("QPS: %lf\n", query_num / diff.count());

  int cnt = 0;
  for (int i = 0; i < query_num; ++i) {
    std::set<int> st(gt + i * gt_k, gt + i * gt_k + K);
    for (auto x : res[i]) {
      if (st.count(x)) cnt++;
    }
  }
  printf("Recall: %lf\n", double(cnt) / query_num / K);


  return 0;
}
