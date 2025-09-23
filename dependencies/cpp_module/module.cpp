#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <fstream>

namespace py=pybind11;

float ray_tri_intersect(const Eigen::Vector3f &ray_orig, const Eigen::Vector3f &ray_dir, 
                        const std::vector<Eigen::Vector3f>& tri, float epsilon=1e-8){

    if (tri.size() < 3) {
        std::cerr << "Error: Triangle has less than 3 vertices!" << std::endl;
        return -1.0;
    }
    Eigen::Vector3f v0v1 = tri[1] - tri[0];
    Eigen::Vector3f v0v2 = tri[2] - tri[0];
    Eigen::Vector3f tri_normal = v0v1.cross(v0v2);
    
    // check if ray and plane are parallel
    float normal_dot_dir = tri_normal.dot(ray_dir);
    if (fabs(normal_dot_dir) < epsilon){
        return -1.0;
    }

    // compute D in Ax+By+Cz+D=0
    float d = -tri_normal.dot(tri[0]);

    // compute t in P = orig + t * dir
    float t = -(tri_normal.dot(ray_orig) + d) / normal_dot_dir;
    if (t < 0 || t > 1){
        return -1.0;
    }
    
    Eigen::Vector3f p = ray_orig + t * ray_dir;
    // inside-outside test
    Eigen::Vector3f edge, vp, new_normal;
    bool flag = true;
    for(int i = 0; i < 3; ++i){
        edge = tri[(i + 1) % 3] - tri[i];
        edge.normalize();
        vp = p - tri[i];
        vp.normalize();
        new_normal = edge.cross(vp);
        // if it has different direction with triangle's normal
        // then p is outside of triangle
        if (tri_normal.dot(new_normal) <= epsilon){
            flag = false;
            break;
        }
    }
    return flag ? t : -1.0;
}

std::map<int, std::vector<float>> get_dict(py::dict dict_input){
    std::map<int, std::vector<float>> out_map;
    for (auto item : dict_input) {
        int key = item.first.cast<int>();
        std::vector<float> value = item.second.cast<std::vector<float>>();
        out_map[key] = value;
    }
    return out_map;
}

template <typename T>
std::vector<T> get_val_1d(py::array_t<T> val_input) {
    py::buffer_info val_array = val_input.request();
    size_t size = val_array.size;

    std::vector<T> val_output;
    for (size_t i = 0; i < size; ++i) {
        val_output.push_back(((T*)val_array.ptr)[i]);
    }
    return val_output;
}

template <typename T>
std::vector<std::vector<T>> get_val_2d(py::array_t<T> val_input) {
    py::buffer_info val_array = val_input.request();
    if (val_array.ndim != 2)
        throw std::runtime_error("get_val_2d(): Number of dimensions must be two");

    int rows = val_array.shape[0];
    int cols = val_array.shape[1];
    auto ptr = static_cast<T*>(val_array.ptr);

    std::vector<std::vector<T>> val_output(rows, std::vector<T>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j){
            val_output[i][j] = ptr[i * cols + j];
        }
    }
    return val_output;
}

std::vector<Eigen::Vector3f> get_verts(py::array_t<float> val_input){
    py::buffer_info val_array = val_input.request();
    size_t size = val_array.size;
    std::vector<Eigen::Vector3f> val_output;
    for (int i = 0; i < int(size / 3); ++i){
        Eigen::Vector3f pos;
        pos << ((float*)val_array.ptr)[3*i+0], ((float*)val_array.ptr)[3*i+1], ((float*)val_array.ptr)[3*i+2];
        val_output.push_back(pos);
    }
    return val_output;
}

std::vector<Eigen::VectorXi> get_faces(py::array_t<int> val_input, int vert_per_face){
    py::buffer_info val_array = val_input.request();
    size_t size = val_array.size;
    Eigen::VectorXi face(vert_per_face);

    std::vector<Eigen::VectorXi> val_output;
    for (int i = 0; i < int(size / vert_per_face); ++i){
        for(int j = 0; j < vert_per_face; ++j){
            face[j] = ((int*)val_array.ptr)[vert_per_face * i + j];
        }
        val_output.push_back(face);
    }
    return val_output;
}

py::array_t<float> get_joint_conf(
    py::array_t<float> joints_input,
    py::array_t<float> cam_input,
    py::array_t<float> verts_input, 
    py::array_t<int> faces_input, 
    py::array_t<bool> verts_visibility_input,
    py::array_t<bool> verts_region_input,
    py::array_t<float> verts_weight_input
){
    
    // std::cout << "start get_joint_vis \n" << std::flush;
    std::vector<Eigen::Vector3f> joints = get_verts(joints_input);
    // camera
    std::vector<Eigen::Vector3f> cam_pos = get_verts(cam_input);
    // vertices positions
    std::vector<Eigen::Vector3f> verts = get_verts(verts_input);
    // faces
    std::vector<Eigen::VectorXi> faces = get_faces(faces_input, 3);
    
    std::vector<bool> verts_visibility = get_val_1d<bool>(verts_visibility_input);
    std::vector<std::vector<bool>> verts_region = get_val_2d<bool>(verts_region_input);
    std::vector<std::vector<float>> verts_weight = get_val_2d<float>(verts_weight_input);

    int j_size = joints.size(); 
    if (j_size != 144){
        std::cerr << "Error: number of joints must be 144!" << std::endl;
    }
    int v_size = verts.size();
    int f_size = faces.size();
    std::vector<float> out_joints_conf(j_size, 0.0);
    
    Eigen::Vector3f ray_orig, ray_dir, p;
    std::vector<Eigen::Vector3f> tri(3);
    float t = -1.0;
    bool visible_flag = true;

    // check joints(determined by skinning weights) visibility 
    // std::cout << "check joints visibility:" << j_size << "," << f_size << std::endl;
    for(int ji = 0; ji < 25; ++ji){
        // std::cout << "================ " << ji << std::endl;
        // std::cout << joints[ji] << std::endl;
        ray_orig = joints[ji];
        ray_dir = cam_pos[0] - ray_orig;
        visible_flag = true;
        std::vector<float> joint_verts_weight = verts_weight[ji];
        std::vector<bool> joint_verts_region = verts_region[ji];
        Eigen::VectorXi face_min(3);
        float t_min = 1000;
        
        // std::cout << i << "," << verts[i] << "\n"<< std::flush;
        std::vector<float> t_arr;
        for(int fi = 0; fi < f_size; ++fi){
            Eigen::VectorXi f = faces[fi];
            if (f[0] >= v_size || f[1] >= v_size || f[2] >= v_size) {
                std::cerr << "Error: Face index out of bounds!" << std::endl;
                continue;
            }
            tri[0] = verts[f[0]];
            tri[1] = verts[f[1]];
            tri[2] = verts[f[2]];
            
            t = ray_tri_intersect(ray_orig, ray_dir, tri);
            if (t < 0){
                continue;
            }
            
            // if current joint is blocked by an invisible triangle, then it is invisible
            if( (!verts_visibility[f[0]]) && (!verts_visibility[f[1]]) && (!verts_visibility[f[2]]) ){
                visible_flag = false;
                break; 
            }
            // if current joint is blocked by a triangle belongs to another part, then it is invisible
            // jaw, leye, reye
            if (ji >= 22 && ji < 25){
                if ( (!joint_verts_region[f[0]]) && (!joint_verts_region[f[1]]) && (!joint_verts_region[f[2]]) ){
                    visible_flag = false;
                    break; 
                }
            }           
            t_arr.push_back(t);
            if (t_arr.size() > 1){
                visible_flag = false;
                break;
            }
            if (t < t_min){
                t_min = t;
                face_min = f;
                visible_flag = true;
            }
        }
        // if (t_arr.size() > 1){
        //     visible_flag = false;
        // }
        if( visible_flag ){
            out_joints_conf[ji] = 1.0;
            // if ( t_min > 1.0 ){
            //     out_joints_conf[ji] = 1.0;
            // }
            // else{
            //     int vert_vis_cnt = 0;
            //     for (int i = 0; i < 3; ++i){
            //         if (verts_visibility[face_min[i]]){
            //             vert_vis_cnt += 1;
            //         }
            //     }
            //     out_joints_conf[ji] = vert_vis_cnt / 3.0;
            // }     
        }
    }

    // left hands use left wrist's visibility
    for (int ji = 25; ji < 40; ++ji){
        out_joints_conf[ji] = out_joints_conf[20];
    }
    // right hands use right wrist's visibility
    for (int ji = 40; ji < 55; ++ji){
        out_joints_conf[ji] = out_joints_conf[21];
    }
    // check other joints (selected from vertices) visibility
    for(int ji = 55; ji < j_size; ++ji){
        ray_orig = joints[ji];
        ray_dir = cam_pos[0] - ray_orig;
        visible_flag = true;

        for(Eigen::VectorXi f : faces){
            tri[0] = verts[f[0]];
            tri[1] = verts[f[1]];
            tri[2] = verts[f[2]];
            t = ray_tri_intersect(ray_orig, ray_dir, tri);
            if(t > 0){
                visible_flag = false;
                break;             
            }
        }
        if(visible_flag){
            out_joints_conf[ji] = 1.0;
        }        
    }
    // std::cout << "check joints visibility finish" << std::endl;
    auto result = py::array_t<float>(out_joints_conf.size());
    auto result_buffer = result.request();
    float *result_ptr = (float *)result_buffer.ptr;
    if (result_buffer.size != out_joints_conf.size()) {
        std::cerr << "Error: Result buffer size mismatch!" << std::endl;
        return result;
    }

    std::memcpy(result_ptr, out_joints_conf.data(), out_joints_conf.size()*sizeof(float));
    return result;

}


py::array_t<float> get_hand_joint_conf(
    py::array_t<float> joints_input,
    py::array_t<float> cam_input,
    py::array_t<float> verts_input, 
    py::array_t<int> faces_input, 
    py::array_t<bool> verts_visibility_input
){
    
    // std::cout << "start get_joint_vis \n" << std::flush;
    std::vector<Eigen::Vector3f> joints = get_verts(joints_input);
    // camera
    std::vector<Eigen::Vector3f> cam_pos = get_verts(cam_input);
    // vertices positions
    std::vector<Eigen::Vector3f> verts = get_verts(verts_input);
    // faces
    std::vector<Eigen::VectorXi> faces = get_faces(faces_input, 3);
    
    std::vector<bool> verts_visibility = get_val_1d<bool>(verts_visibility_input);
    int j_size = joints.size(); 
    int v_size = verts.size();
    int f_size = faces.size();
    std::vector<float> out_joints_conf(j_size, 0.0);
    
    Eigen::Vector3f ray_orig, ray_dir, p;
    std::vector<Eigen::Vector3f> tri(3);
    float t = -1.0;
    bool visible_flag = true;

    // check joints(determined by skinning weights) visibility 
    // std::cout << "check joints visibility:" << j_size << "," << f_size << std::endl;
    for(int ji = 0; ji < j_size; ++ji){
        // std::cout << "================ " << ji << std::endl;
        // std::cout << joints[ji] << std::endl;
        ray_orig = joints[ji];
        ray_dir = cam_pos[0] - ray_orig;
        visible_flag = true;
        Eigen::VectorXi face_min(3);
        float t_min = 1000;
        
        // std::cout << i << "," << verts[i] << "\n"<< std::flush;
        std::vector<float> t_arr;
        for(int fi = 0; fi < f_size; ++fi){
            Eigen::VectorXi f = faces[fi];
            if (f[0] >= v_size || f[1] >= v_size || f[2] >= v_size) {
                std::cerr << "Error: Hand index out of bounds!" << std::endl;
                continue;
            }
            tri[0] = verts[f[0]];
            tri[1] = verts[f[1]];
            tri[2] = verts[f[2]];
            
            t = ray_tri_intersect(ray_orig, ray_dir, tri);
            if (t < 0){
                continue;
            }
            // if current joint is blocked by an invisible triangle, then it is invisible
            if( (!verts_visibility[f[0]]) && (!verts_visibility[f[1]]) && (!verts_visibility[f[2]]) ){
                visible_flag = false;
                break; 
            }
                     
            t_arr.push_back(t);
            if (t_arr.size() > 1){
                visible_flag = false;
                break;
            }
            if (t < t_min){
                t_min = t;
                face_min = f;
                visible_flag = true;
            }
        }
        if( visible_flag ){
            out_joints_conf[ji] = 1.0; 
        }
    }

    // std::cout << "check joints visibility finish" << std::endl;
    auto result = py::array_t<float>(out_joints_conf.size());
    auto result_buffer = result.request();
    float *result_ptr = (float *)result_buffer.ptr;
    if (result_buffer.size != out_joints_conf.size()) {
        std::cerr << "Error: Result buffer size mismatch!" << std::endl;
        return result;
    }

    std::memcpy(result_ptr, out_joints_conf.data(), out_joints_conf.size()*sizeof(float));
    return result;

}

PYBIND11_MODULE(get_vis, m){
    m.doc() = "cpp module";
    m.def("get_joint_conf", &get_joint_conf, "get_joint_conf");
    m.def("get_hand_joint_conf", &get_hand_joint_conf, "get_hand_joint_conf");
}