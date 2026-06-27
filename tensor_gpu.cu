#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

class Shape{
public:
    size_t C=1,T=1,B=1,dim=1;
    Shape(size_t dim1):C(dim1),dim(1){}
    Shape(size_t dim1,size_t dim2):C(dim1),T(dim2),dim(2){}
    Shape(size_t dim1,size_t dim2,size_t dim3):C(dim1),T(dim2),B(dim3),dim(3){}
    size_t tot_ele() const { return C*T*B; }
    size_t get_dim(){ return dim;}
};

enum class DataType{ Int, Float, Double};
enum class Device{ CPU,GPU};

template<typename T>
class Tensor{
private:
    Shape shape;
    size_t size;
    Device device;
    T* cpu_data=nullptr;
    T* gpu_data=nullptr;
    DataType dtype;
    void assign_data_type(){
        if(is_same_v<T,int>) dtype=DataType::Int;
        else if(is_same_v<T,float>) dtype=DataType::Float;
        else if(is_same_v<T,double>) dtype=DataType::Double;
    }
public:
    Tensor(Shape s,Device new_device=Device::CPU):shape(s),size(s.tot_ele()),device(new_device){
        assign_data_type();
        if(device==Device::CPU){
            cpu_data=new T[size]{};
        }
        else{
            cudaError_t err=cudaMalloc(&gpu_data,size*sizeof(T));
            if(err!=cudaSuccess){
                throw runtime_error(string("The memory allocation in GPU has faled.")+cudaGetErrorString(err));
            }
            cudaMemset(gpu_data,0,size*sizeof(T));
        }
    }
    ~Tensor(){
        delete[] cpu_data;
        cpu_data=nullptr;
        if(gpu_data){
            cudaFree(gpu_data);
            gpu_data=nullptr;
        }
    }
    void add_data(vector<T> arr){
        if(device==Device::CPU){
            if(size!=arr.size()){
                throw runtime_error("The sizes of input and tensor are not equal.\n");
            }
            else{
                for(size_t i=0;i<size;i++){
                    cpu_data[i]=arr[i];
                }
                cout<<"Tensor is created on cpu successfully.\n";
            }
        }
        else{
            cudaMemcpy(gpu_data,arr.data(),size*sizeof(T),cudaMemcpyHostToDevice);
            cout<<"Tensor is created on gpu successfully.\n";
        }
    }
    void print(){
        T* dummy_ptr=nullptr;
        bool flag=false;
        if(device==Device::CPU){
            dummy_ptr=cpu_data;
        }
        else{
            dummy_ptr=new T[size]{};
            cudaMemcpy(dummy_ptr,gpu_data,size*sizeof(T),cudaMemcpyDeviceToHost);
            flag=true;
        }
        for(size_t i=0;i<shape.B;i++){
            if(shape.dim==3) 
            cout<<"Batch"<<i<<endl;
            for(size_t j=0;j<shape.T;j++){
                cout<<"[ ";
                for(size_t k=0;k<shape.C;k++){
                    size_t idx=i*shape.T*shape.C+j*shape.C+k;
                    cout<<dummy_ptr[idx]<< (k==shape.C-1 ? "":", ");
                }
                cout<<"]\n";
            }
        }
        if(flag)
        delete[] dummy_ptr;
    }
    void to(Device new_device){
        if(new_device==device) return;
        if(new_device==Device::CPU){
            cpu_data=new T[size]{};
            cudaMemcpy(cpu_data,gpu_data,size*sizeof(T),cudaMemcpyDeviceToHost);
            cudaFree(gpu_data);
            gpu_data=nullptr;
        }
        else{
            cudaMalloc(&gpu_data,size*sizeof(T));
            cudaMemcpy(gpu_data,cpu_data,size*sizeof(T),cudaMemcpyHostToDevice);
            delete[] cpu_data;
            cpu_data=nullptr;
        }
        device=new_device;
    }
    Shape get_shape(){return shape;}
    size_t get_size(){return size;}
    Device get_device(){return device;}
    T* get_cpu_data(){return cpu_data;}
    T* get_gpu_data(){return gpu_data;}
    DataType get_dtype(){return dtype;}
};

int main(){
    Shape s(2,2,2);
    vector<int> arr={1,2,3,4,5,6,7,8};
    Tensor<int> t1(s);
    t1.add_data(arr);
    Tensor<int> t2(s,Device::GPU);
    t2.add_data(arr);
    t1.to(Device::GPU);
    //t1.print();
    t2.print();
}