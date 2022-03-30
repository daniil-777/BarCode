#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <boost/pending/disjoint_sets.hpp>

namespace sublevel {
typedef std::vector<std::vector<int>> Graph;

struct Ans {
    Ans(int e, int d, int es, int ds) 
        : eat(e), death(d), eat_size(es), death_size(ds), 
        total_height(0.0) {
    }
    int eat;
    int death;
    int eat_size;
    int death_size;
    float total_height;
};

class RootedForest {
public:
    Graph tree_;
    std::vector<int> pred_;
    std::vector<int> height_;

    explicit RootedForest(Graph tree) : tree_(tree) {
    }

    void Initilize() {
        pred_.resize(tree_.size(), -1);
        height_.resize(tree_.size(), -1);
        std::queue<int> que;
        size_t counter = 0;
        while (counter < tree_.size()) {
            if (que.empty()) {
                int next_start = 0;
                while (height_[next_start] != -1) {
                    ++next_start;
                }
                que.push(next_start);
                ++counter;
                height_[next_start] = 0;
            }
            while (!que.empty()) {
                int current = que.front();
                que.pop();
                for (auto neu : tree_[current]) {
                    if (height_[neu] == -1) {
                        height_[neu] = height_[current] + 1;
                        pred_[neu] = current;
                        que.push(neu);
                        ++counter;
                    }
                }
            }
        }
    }

    bool GetWay (int lhs, int rhs, std::vector<int>& answer) {
        std::vector<int> left_way;
        std::vector<int> right_way;
        while (height_[lhs] > height_[rhs]) {
            left_way.push_back(lhs);
            lhs = pred_[lhs];
        }
        while (height_[lhs] < height_[rhs]) {
            right_way.push_back(rhs);
            rhs = pred_[rhs];
        }
        while ((lhs != -1 && rhs != -1) && lhs != rhs) {
            left_way.push_back(lhs);
            lhs = pred_[lhs];
            right_way.push_back(rhs);
            rhs = pred_[rhs];                        
        }
        if (lhs != -1 && rhs != -1) {
            for (size_t step = 0; step < left_way.size(); ++step) {
                answer.push_back(left_way[step]);
            }
            answer.push_back(lhs);
            for (size_t step = right_way.size(); step > 0; --step) {
                answer.push_back(right_way[step - 1]);
            }
            return true;
        }
        return false;
    }
};

class VirtualCloud { 
public:
    float* values_;
    size_t cloud_size_;
    std::vector<int> order_;
    std::set<int> trash_;
    Graph graph_;
    Graph minima_graph_;
    VirtualCloud(float* val, size_t size) : values_(val), cloud_size_(size) {
    }
    virtual ~VirtualCloud() = default;
    void GetOrder() {
        order_.reserve(cloud_size_);
        for (int index = 0; index < static_cast<int>(cloud_size_); ++index) {
            order_.push_back(index);
        }
        std::sort(order_.begin(), order_.end(), 
                  [&](int lhs, int rhs) { return (*(values_ + lhs) < *(values_ + rhs) ||
                                                  (*(values_ + lhs) == *(values_ + rhs) && lhs < rhs)); } );
    }

    virtual void GetGraph() {
    }

    virtual void SetGraph(Graph&& gr) {
        graph_ = gr;
    }

    std::map<int, Ans> SublevelHomology() {
        GetGraph();
        minima_graph_.resize(cloud_size_);
        std::map<int, Ans> answer;
        int lenght = graph_.size();
        std::vector<int> back_ord(lenght, 0);
        for (int id = 0; id < lenght; ++id) {
            back_ord[order_[id]] = id;
        }
        std::vector<int> siz(lenght + 1, 0);
        std::vector<int> ord(lenght + 1, 0);
        boost::disjoint_sets<int*,int*> ds(&ord[0], &siz[0]);
        auto comp = [&](int lhs, int rhs){ return back_ord[lhs - 1] < back_ord[rhs - 1]; };
        for (int ind = 0; ind < lenght; ++ind) {
            int vertex = order_[ind];
            if (trash_.find(vertex) != trash_.end()) {
                continue;
            }
            std::map<int, int, decltype(comp)> clusters(comp);
            for (auto neubour : graph_[vertex]) {
                if (trash_.find(neubour) == trash_.end()) {
                    int next_cluster = ds.find_set(neubour + 1);
                    if (clusters.find(next_cluster) == clusters.end()) {
                        clusters.emplace(next_cluster, neubour);
                    } else if (back_ord[neubour] < back_ord[clusters[next_cluster]]) { 
                        clusters[next_cluster] = neubour;
                    }
                }        
            }
            for (auto& item : clusters) {
                minima_graph_[vertex].push_back(item.second);
                minima_graph_[item.second].push_back(vertex);
            }
            if (clusters.empty()) {
                ds.make_set(vertex + 1);
                answer.emplace(vertex, Ans(-1, -1, -1, 1));
                continue;
            }
            int start = clusters.begin()->first;
            ds.make_set(vertex + 1);
            ds.link(vertex + 1, start);
            ++answer.at(start - 1).death_size;
            answer.at(start - 1).total_height += *(values_ + vertex) - *(values_ + start - 1);
            for (auto it = next(clusters.begin()); it != clusters.end(); ++it) {
                ord[start] = std::max(ord[start], ord[it->first]);
                ds.link(it->first, start);
                answer.at(it->first - 1).eat = start - 1;
                answer.at(it->first - 1).death = vertex;
                answer.at(it->first - 1).eat_size = answer.at(start - 1).death_size;
                answer.at(start - 1).death_size += answer.at(it->first - 1).death_size;
                answer.at(start - 1).total_height += answer.at(it->first - 1).total_height + 
                                                     (*(values_ + it->first - 1) - *(values_ + start - 1)) * 
                                                     answer.at(it->first - 1).death_size;
            }
        }
        return answer;
    }
};

class GridCloud : public VirtualCloud {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> shift_;

    int GetIndex(int point, int dim) {
        return (point % shift_[dim + 1]) / shift_[dim];
    }

public:
    GridCloud(float* val, size_t size, std::vector<size_t> sh) 
        : VirtualCloud(val, size), shape_(sh) {
        shift_.resize(shape_.size() + 1, 1);
        for (size_t id = 1; id < shape_.size() + 1; ++id) {
            shift_[id] = shift_[id - 1] * shape_[id - 1];
        }
    }

    void GetGraph() {
        graph_.resize(cloud_size_);
        for (size_t index = 0; index < cloud_size_; ++index) {
            for (size_t dim = 0; dim < shape_.size(); ++dim) {
                size_t down = index - shift_[dim];
                if (GetIndex(index, dim) != 0 && *(values_ + index) >= *(values_ + down)) {
                    graph_[index].push_back(down);
                }
                size_t up = index + shift_[dim];
                if (GetIndex(up, dim) != 0 && *(values_ + index) > *(values_ + up)) {
                    graph_[index].push_back(up);
                }
            }            
        }
    }
};

}

