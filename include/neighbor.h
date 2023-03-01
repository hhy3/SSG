//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <pthread.h>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

#include "util.h"

namespace efanna2e {

struct Neighbor {
  unsigned id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(unsigned id, float distance, bool f = true)
      : id{id}, distance{distance}, flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

typedef std::lock_guard<std::mutex> LockGuard;
struct nhood {
  // std::mutex lock;
  pthread_mutex_t lock;
  std::vector<Neighbor> pool;
  unsigned M;

  std::vector<unsigned> nn_old;
  std::vector<unsigned> nn_new;
  std::vector<unsigned> rnn_old;
  std::vector<unsigned> rnn_new;

  nhood() {}
  nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
    M = s;
    nn_new.resize(s * 2);
    GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
    nn_new.reserve(s * 2);
    pool.reserve(l);
    pthread_mutex_init(&lock, NULL);
  }

  nhood(const nhood &other) {
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(),
              std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
  }
  void insert(unsigned id, float dist) {
    // LockGuard guard(lock);
    pthread_mutex_lock(&lock);
    if (dist > pool.front().distance) {
      pthread_mutex_unlock(&lock);
      return;
    }
    for (unsigned i = 0; i < pool.size(); i++) {
      if (id == pool[i].id) {
        pthread_mutex_unlock(&lock);
        return;
      }
    }
    if (pool.size() < pool.capacity()) {
      pool.push_back(Neighbor(id, dist, true));
      std::push_heap(pool.begin(), pool.end());
    } else {
      std::pop_heap(pool.begin(), pool.end());
      pool[pool.size() - 1] = Neighbor(id, dist, true);
      std::push_heap(pool.begin(), pool.end());
    }
    pthread_mutex_unlock(&lock);
  }

  template <typename C>
  void join(C callback) const {
    for (unsigned const i : nn_new) {
      for (unsigned const j : nn_new) {
        if (i < j) {
          callback(i, j);
        }
      }
      for (unsigned j : nn_old) {
        callback(i, j);
      }
    }
  }
};

struct LockNeighbor {
  // std::mutex lock;
  pthread_mutex_t lock;
  std::vector<Neighbor> pool;
};

struct SimpleNeighbor {
  unsigned id;
  float distance;

  SimpleNeighbor() = default;
  SimpleNeighbor(unsigned id_, float distance_)
      : id(id_), distance(distance_) {}

  inline bool operator<(const SimpleNeighbor &other) const {
    return distance < other.distance;
  }
};

struct SimpleNeighbors {
  std::vector<SimpleNeighbor> pool;
};

static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance)
      right = mid;
    else
      left = mid;
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
  memmove((char *)&addr[right + 1], &addr[right],
          (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}
class NeighborSet {
 public:
  explicit NeighborSet(size_t capacity = 0)
      : size_(0), capacity_(capacity), data_(capacity_ + 1) {}

  void insert(SimpleNeighbor nbr) {
    if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
      return;
    }
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (data_[mid].distance > nbr.distance) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
    data_[lo] = {nbr.id, nbr.distance, true};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
  }

  int pop() {
    data_[cur_].flag = false;
    size_t pre = cur_;
    while (cur_ < size_ && !data_[cur_].flag) {
      cur_++;
    }
    return data_[pre].id;
  }

  bool has_next() const { return cur_ < size_; }

  int id(int i) { return data_[i].id; }

  std::vector<int> get_topk(int k) {
    std::vector<int> ans(k);
    for (int i = 0; i < k; ++i) {
      ans[i] = data_[i].id;
    }
    return ans;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  Neighbor &operator[](size_t i) { return data_[i]; }

  const Neighbor &operator[](size_t i) const { return data_[i]; }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

 private:
  size_t size_, capacity_, cur_;
  std::vector<Neighbor> data_;
};

}  // namespace efanna2e

#endif  // EFANNA2E_GRAPH_H
