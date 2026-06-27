#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

class Shape{
public:
    size_t d1=1;
    size_t d2=1;
    size_t d3=1;
    size_t dims=1;
    Shape(size_t dim1): d1(dim1),dims(1){}
    Shape(size_t dim1,size_t dim2): d1(dim1),d2(dim2),dims(2){}
    Shape(size_t dim1,size_t dim2,size_t dim3): d1(dim1),d2(dim2),d3(dim3),dims(3){}
    size_t tot_ele() const{
        return d1*d2*d3;
    }
};

enum class DataType{
    Int,
    Float,
    Double
};

template<typename T>
class Tensor{
private:
    Shape shape;
    DataType dtype;
    T* data;
    size_t size;
    void assign_data_type(){
        if(is_same_v<T,int>) dtype=DataType::Int;
        else if(is_same_v<T,float>) dtype=DataType::Float;
        else if(is_same_v<T,double>) dtype=DataType::Double;
        else throw runtime_error("The data type is not supported.\n");
    }
public:
    Tensor(Shape s): shape(s),size(s.tot_ele()){
        assign_data_type();
        data=new T[size] {};
    }
    void add_data(vector<T> arr){
        if(size!=arr.size()) 
        throw runtime_error("The size of the input and data are not equal.");
        for(size_t i=0;i<size;i++){
            data[i]=arr[i];
        }
    }
    ~Tensor(){
        delete[] data;
    }
    Shape get_shape(){ return shape;}
    DataType get_data_type(){ return dtype;}
    T* get_data(){ return data;}
    size_t get_size(){ return size;}
};
int main(){
    try{
        Shape s(2,1,2);
        Tensor<int> t1(s);
        t1.add_data({1,0,0,1});
        cout<<"Tensor created successfully\n";
    }
    catch(runtime_error const &e){
        cerr<<"Exception occurred:"<<e.what()<<endl;
    }
    return 0;
}