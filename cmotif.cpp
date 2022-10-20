#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include <typeinfo>
#include <utility>
#include <assert.h> 
#include "progress_bar.hpp"
#include <thread>
#include <mutex>
using namespace std;

mutex mtx;

unordered_map<int, unordered_set<int>> intersection_cache;
unordered_map<int, vector<int>> v12_cache;
inline int make_key(int i,int j) {
    return (max(i,j)*(max(i,j)+1))/2+min(i,j);
}

inline void get_latest_increment(vector<vector<int>>& motif2distribution, vector<int>& latest_increment){
    for (int i=3; i!=13; ++i){
        latest_increment.push_back(motif2distribution[i].back());
    }
}


inline bool intersect(vector<int>& a, vector<int>& b, vector<int> exclude){
    int size_a = a.size();
    int size_b = b.size(); // assert a and b are sorted already
    sort(exclude.begin(), exclude.end());
    int v_size = min(size_a, size_b);
    vector<int> v(v_size);
    vector<int>::iterator v_end=set_intersection(a.begin(), a.end(), b.begin(), b.end(), v.begin());
    vector<int> v2(v_size);
    vector<int>::iterator v2_end=set_difference(v.begin(), v_end, exclude.begin(), exclude.end(), v2.begin());
    return v2_end - v2.begin() > 0;
}

inline bool intersects(const vector<unordered_set<int>>& H, int a_i, int b_i, int v1, int v2 =-1){

    int key = make_key(a_i, b_i);
    unordered_map<int, unordered_set<int>>::iterator it = intersection_cache.find(key);
    if (it != intersection_cache.end()){
        unordered_set<int>& intersection = it -> second;
        unsigned int has1 = (intersection.find(v1) != intersection.end());
        unsigned int has2 = (v2 != -1) && (intersection.find(v2) != intersection.end());
        return intersection.size() > (has1 + has2);    
    }
    else{
        const unordered_set<int>& a = H.at(a_i);
        const unordered_set<int>& b = H.at(b_i);
        unordered_set<int> intersection;
        if (a.size() > b.size()){
            for (int e: b){
                if (a.find(e) != a.end()){
                    intersection.insert(e);
                }
            }
        }
        else{
            for (int e: a){
                if (b.find(e) != b.end()){
                    intersection.insert(e);
                }
            }
        }
        intersection_cache[key] = intersection;

        unsigned int has1 = intersection.find(v1) != intersection.end();
        unsigned int has2 = (v2 != -1) && intersection.find(v2) != intersection.end();
        return intersection.size() > (has1 + has2); 
    }
       
}


inline bool contain(const vector<int>& v, int x){
    return find(v.begin(), v.end(), x) != v.end();
}

inline bool contains(const unordered_set<int>& v, int x){
    return (v.find(x) != v.end());
}


inline void union_(const vector<int>& a, const vector<int>& b, vector<int>& out_vector){
    vector<int>::iterator it = set_union(a.begin(), a.end(), b.begin(), b.end(), out_vector.begin());
    out_vector.resize(it-out_vector.begin());
}


inline bool arr_equal(bool a[], int b[], int size=5){
    for (int i=0; i!= size; i++){
        if (a[i] != bool(b[i])){
            return false;
        }
    }
    return true;
}

inline void read_data(string fname, vector<vector<int>>& content){
	vector<int> row;
	string line, word;
 
	fstream file(fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str_(line);
 
			while(getline(str_, word, ','))
				row.push_back(stoi(word));
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the file\n";
}

inline void read_data2(string fname, vector<unordered_set<int>>& content){
	unordered_set<int> row;
	string line, word;
 
	fstream file(fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str_(line);
 
			while(getline(str_, word, ','))
				row.insert(stoi(word));
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the file\n";
}

inline void write_data(string fname, const vector<vector<float>>& feature_mat){
    fstream file(fname, ios::out);
	if(file.is_open()){
        for (const vector<float> feature: feature_mat){
            for (const float value: feature){
                file << to_string(value) << ",";
            }
            file << "\n";
        }
    }
    else{
        cout<<"Could not open the file\n";
    }
}
inline void get_node_he_neighbors(const vector<unordered_set<int>>H, unordered_map<int, vector<int>>& node2hes, unordered_map<int, unordered_set<int>>& node2nodes){
    int he_index = 0;
    for (const unordered_set<int>& he: H){
        for (const int node: he){
            node2hes[node].push_back(he_index);
        }
        he_index++;
    }
    for (auto& pair: node2hes){
        const int node = pair.first;
        sort(pair.second.begin(), pair.second.end());
        for (int he_i: pair.second){
            node2nodes[node].insert(H[he_i].begin(), H[he_i].end());
            node2nodes[node].erase(node);
        }
    }
}

inline float* compute_stats(vector<int> arr){
    float s= 0;
    float min = 1e5;
    float max = -1;
    for (int e: arr){
        s+= e;
        if (e<min) min=e;
        if (e>max) max=e;
    }
    float mean = s/arr.size();
    s = 0;
    for (int e: arr){
        s += (e-mean)*(e-mean);
    }
    float std = sqrt(s/arr.size());
    float* result = new float[4]{min, max, mean, std};

    return result;
}

inline void vectorize_distribution(vector<vector<int>>& motif2distribution, vector<float>& out_feature){
    for (vector<int> distribution: motif2distribution){
        float* stats = compute_stats(distribution);
        for (int i=0; i<4;i++){
            float stat = stats[i];
            out_feature.push_back(stat);
        }
        delete[] stats;
    }

}

inline void print(int i, char end_='\n'){
    cout << i << end_;
}

inline void rand_fill(vector<int>& v, int max_){
    for (vector<int>::iterator it=v.begin(); it!=v.end(); ++it){
        *it = rand() % max_;
    }
}


inline void upscale_distribution(vector<vector<int>>& motif2distribution, const int& downsample, const int& hes_size){
    for (int i=3; i!=13; ++i){
        int count = motif2distribution[i].back();
        motif2distribution[i].back() = count * hes_size * (hes_size-1) / (2 * downsample);
    }
}


inline void compute_feature(int clique_i, const vector<int>& clique, const unordered_map<int, vector<int>>& node2hes, \
const unordered_map<int, unordered_set<int>>& node2nodes, const vector<unordered_set<int>>& H, vector<float>& out_feature, int& downsample){
    vector<vector<int>> motif2distribution(13);
    int deg = 0;
    for (int v: clique){
        const vector<int>& hes = node2hes.at(v);
        deg += node2nodes.at(v).size();
        for (int motif_i: {1,2,3}){
            motif2distribution[motif_i-1].push_back(0);
        }
        motif2distribution[1-1].back() = hes.size();
        for (int he1_i=0; he1_i<hes.size(); he1_i++){
            for (int he2_i=he1_i+1; he2_i<hes.size(); he2_i++){
                int he1 = hes[he1_i];
                int he2 = hes[he2_i];
                bool flag = !intersects(H, he1, he2, v);
                motif2distribution[2-1].back() += flag;
                motif2distribution[3-1].back() += !flag;
            }
        }
    }
    int clique_size = clique.size();
    if (clique_size == 1){
        for (int motif_i=4; motif_i<14; motif_i++){
            motif2distribution[motif_i-1].push_back(0);
        }
    }
    else{
        for (int v1_i=0; v1_i<clique_size; ++v1_i){
            int v1 = clique[v1_i];
            for (int v2_i=v1_i+1; v2_i<clique_size; ++v2_i){
                // check cache for this edge (v1, v2)
                int v2 = clique[v2_i];
                int key_v12 = make_key(v1, v2);
                unordered_map<int, vector<int>>::iterator it = v12_cache.find(key_v12);
                // cache found!
                if (it != v12_cache.end()){
                    const vector<int>& latest_increment = it->second;
                    for (int motif_i=4; motif_i<14; motif_i++){
                        motif2distribution[motif_i-1].push_back(latest_increment[motif_i-4]);
                    }
                    continue;
                }
                // cache not found
                for (int motif_i=4; motif_i<14; motif_i++){
                    motif2distribution[motif_i-1].push_back(0);
                }
                const vector<int>& hes1 = node2hes.at(v1);
                const vector<int>& hes2 = node2hes.at(v2);
                vector<int> hes(hes1.size()+hes2.size());
                // cout <<"v1 v2"<<v1<<" "<< v2<< " hes size"<<hes.size()<<" intersection_cache size: "<< intersection_cache.size()<<" v12 cache size "<<v12_cache.size();
                union_(hes1, hes2, hes);

                for (int he: hes){
                    bool flag = contains(H.at(he), v1) && contains(H.at(he), v2);
                    motif2distribution[4-1].back() += flag;
                    motif2distribution[5-1].back() += !flag;
                }
                int hes_size = hes.size();
                // cout<<"hes_size"<<hes_size<<(hes_size*hes_size/2/downsample)<<endl;

                if (hes_size > (int)sqrt(downsample) & downsample>0) {
                    vector<int> he_l1(downsample); vector<int> he_l2(downsample);
                    rand_fill(he_l1, hes_size); rand_fill(he_l2, hes_size);
                    for (int ii=0; ii != downsample; ++ii){
                        int he1_i = he_l1[ii]; int he2_i = he_l2[ii]; 
                        int he1_ii = hes[he1_i]; int he2_ii = hes[he2_i]; 
                        const unordered_set<int>& he1 = H[he1_ii];
                        const unordered_set<int>& he2 = H[he2_ii]; //?????
                        bool its = intersects(H, he1_ii, he2_ii, v1, v2);
                        bool c11 = contains(he1, v1);
                        bool c12 = (!c11) || contains(he1, v2);
                        bool c21 = contains(he2, v1);
                        bool c22 = (!c21) || contains(he2, v2);
                        bool flags[5] = {its, c11, c12, c21, c22};
                        // bool flags[5] = {true};
                        int a1[] = {0,1,0,0,1}; int b1[] = {0,0,1,1,0};
                        motif2distribution[6-1].back() += arr_equal(flags, a1) || arr_equal(flags, b1);
                        int a2[] = {0,1,0,1,0}; int b2[] = {0,0,1,0,1};
                        motif2distribution[7-1].back() += arr_equal(flags, a2) || arr_equal(flags, b2);
                        int a3[] = {1,1,0,0,1}; int b3[] = {1,0,1,1,0};
                        motif2distribution[8-1].back() += arr_equal(flags, a3) || arr_equal(flags, b3);
                        motif2distribution[9-1].back() += !flags[0] && (flags[1]+flags[2]+flags[3]+flags[4] == 3);
                        int a4[] = {1,1,0,1,0}; int b4[] = {1,0,1,0,1};
                        motif2distribution[10-1].back() += arr_equal(flags, a4) || arr_equal(flags,b4);
                        motif2distribution[11-1].back() += !flags[0] && (flags[1]&& flags[2]&& flags[3]&& flags[4]);
                        motif2distribution[12-1].back() += flags[0] && (flags[1]+flags[2]+flags[3]+flags[4] == 3);
                        motif2distribution[13-1].back() += flags[0] && flags[1]&& flags[2]&& flags[3]&& flags[4];
                    }
                    upscale_distribution(motif2distribution, downsample, hes_size);
                }
                else{
                    for (int he1_i=0; he1_i!=hes_size; ++he1_i){
                        for (int he2_i=he1_i+1; he2_i!=hes_size; ++he2_i){
                            int he1_ii = hes[he1_i]; int he2_ii = hes[he2_i]; 
                            const unordered_set<int>& he1 = H[he1_ii];
                            const unordered_set<int>& he2 = H[he2_ii]; //?????
                            bool its = intersects(H, he1_ii, he2_ii, v1, v2);
                            bool c11 = contains(he1, v1);
                            bool c12 = (!c11) || contains(he1, v2);
                            bool c21 = contains(he2, v1);
                            bool c22 = (!c21) || contains(he2, v2);
                            bool flags[5] = {its, c11, c12, c21, c22};
                            // bool flags[5] = {true};
                            int a1[] = {0,1,0,0,1}; int b1[] = {0,0,1,1,0};
                            motif2distribution[6-1].back() += arr_equal(flags, a1) || arr_equal(flags, b1);
                            int a2[] = {0,1,0,1,0}; int b2[] = {0,0,1,0,1};
                            motif2distribution[7-1].back() += arr_equal(flags, a2) || arr_equal(flags, b2);
                            int a3[] = {1,1,0,0,1}; int b3[] = {1,0,1,1,0};
                            motif2distribution[8-1].back() += arr_equal(flags, a3) || arr_equal(flags, b3);
                            motif2distribution[9-1].back() += !flags[0] && (flags[1]+flags[2]+flags[3]+flags[4] == 3);
                            int a4[] = {1,1,0,1,0}; int b4[] = {1,0,1,0,1};
                            motif2distribution[10-1].back() += arr_equal(flags, a4) || arr_equal(flags,b4);
                            motif2distribution[11-1].back() += !flags[0] && (flags[1]&& flags[2]&& flags[3]&& flags[4]);
                            motif2distribution[12-1].back() += flags[0] && (flags[1]+flags[2]+flags[3]+flags[4] == 3);
                            motif2distribution[13-1].back() += flags[0] && flags[1]&& flags[2]&& flags[3]&& flags[4];
                        }
                    }

                }
                // update v12 cache: collect lateste increment, write to cache
                vector<int> latest_increment;
                get_latest_increment(motif2distribution, latest_increment);
                v12_cache[key_v12] = latest_increment;
            }
        }
    }
    out_feature.push_back(clique_size);
    out_feature.push_back(float(deg)/float(clique_size));
    vectorize_distribution(motif2distribution, out_feature);
}



void compute_feature_chunk(int start_index, int end_index, \
const unordered_map<int, vector<int>>& node2hes, const unordered_map<int, unordered_set<int>>& node2nodes,\
const vector<vector<int>>& candidates, const vector<unordered_set<int>>& H, \
vector<vector<float>>& feature_mat, progressbar& bar, int& downsample){
    assert(end_index<=candidates.size());
    cout<< "computing candidates: "<< start_index<<" - "<<end_index<<endl;
    for (int i = start_index; i != end_index; ++i){
        compute_feature(i+1, candidates[i], node2hes, node2nodes, H, feature_mat[i], downsample);
        mtx.lock(); bar.update(); mtx.unlock();
    }
}


int main(int argc, char* argv[]){
    // to speed up: use cache + parallel, some motifs don't need to be count, sampling he neighbors
    bool parallel = false;
    int n_threads = 1;
    int downsample = stoi(string(argv[2]));

    string data_dir = "data/"+string(argv[1]) + "/";
    if (argc>3){n_threads = stoi(string(argv[3])); parallel=true;}

    vector<vector<int>> candidates;
    read_data(data_dir+"candidates.csv", candidates); 
    vector<unordered_set<int>> H;
    read_data2(data_dir+"H.csv", H);
    

    unordered_map<int, vector<int>> node2hes;
    unordered_map<int, unordered_set<int>> node2nodes;
    get_node_he_neighbors(H, node2hes, node2nodes);
    
    vector<vector<float>> feature_mat(candidates.size());


    progressbar bar(candidates.size());
    if (!parallel){
        cout<<"sequential execution, 1 thread"<<endl;
        int start_index = 0; int end_index = candidates.size();
        compute_feature_chunk(start_index, end_index, node2hes, node2nodes, candidates,  H, feature_mat, bar, downsample);
    }
    else{
        cout<<"parallel execution ... threads:"<<n_threads<<endl;
        vector<thread> threads;
        int total = candidates.size(); int step_size = total / n_threads + 1; int thread_i = 0;
        for (int start_index=0; start_index < total; start_index += step_size){
            int end_index = min(start_index + step_size, total);
            threads.emplace_back(thread{compute_feature_chunk, start_index, end_index, \
            ref(node2hes), ref(node2nodes), \
            ref(candidates),  ref(H), \
            ref(feature_mat), ref(bar), ref(downsample)});
            cout << "creating thread for candidate " << start_index << " - "<<end_index<<endl;
            ++thread_i;
        }

        for (thread& t: threads){
            cout<<"check one thread"<<endl;
            if (t.joinable()){t.join();}
        }
    }
    write_data(data_dir + "features"+to_string(feature_mat.size())+ ".csv", feature_mat);
    return 0;
}

