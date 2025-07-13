#include <D:\dyplom\praca\eigen\Eigen\Dense> //Ścieżka do frameworku Eigen
#include <D:\dyplom\praca\onnxruntime-1.22.0\include\onnxruntime_cxx_api.h> //Ścieżka do silnika ONNX
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

using namespace Eigen;
using namespace Ort;
using namespace std; // Dodana dyrektywa

int main() {
    // --- Konfiguracja ONNX Runtime ---
    Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Eigen_Test");
    SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Ustaw tryb optymalizacji grafu (opcjonalnie, ale często zalecane)
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // --- Ścieżka do modelu ONNX ---
    const wchar_t* model_path = L"model/model.onnx";

    Session session(nullptr);

    try {
        cout << "Ladowanie modelu z: ";
#ifdef _WIN32
        wcout << model_path << endl;
#else
        cout << model_path << endl;
#endif
        session = Session(env, model_path, session_options);
        cout << "Model ONNX zaladowany pomyslnie." << endl;
    }
    catch (const Ort::Exception& e) {
        cerr << "Blad podczas ladowania modelu ONNX: " << e.what() << endl;
        return 1;
    }


    // --- Pobieranie informacji o wejściach i wyjściach modelu ---
    AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    cout << "Liczba wejsc modelu: " << num_input_nodes << endl;
    cout << "Liczba wyjsc modelu: " << num_output_nodes << endl;

    AllocatedStringPtr input_name_allocated = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_allocated.get();
    cout << "Nazwa wejscia [0]: " << input_name << endl;

    AllocatedStringPtr output_name_allocated = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_allocated.get();
    cout << "Nazwa wyjscia [0]: " << output_name << endl;

    TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
    vector<int64_t> input_dims = input_tensor_info.GetShape();

    cout << "Typ elementu wejscia [0]: " << input_type << endl;
    cout << "Wymiary wejscia [0]: ";
    for (int64_t dim : input_dims) {
        cout << dim << " ";
    }
    cout << endl;


    // --- Przygotowanie danych wejściowych za pomocą Eigen ---
    MatrixXf eigen_input(2, 1);
    eigen_input(0, 0) = 8.0f;
    eigen_input(1, 0) = 9.0f;

    cout << "\nDane wejsciowe (Eigen Matrix):\n" << eigen_input << endl;

    vector<float> input_tensor_values;
    input_tensor_values.reserve(eigen_input.size());
    for (int i = 0; i < eigen_input.rows(); ++i) {
        for (int j = 0; j < eigen_input.cols(); ++j) {
            input_tensor_values.push_back(eigen_input(i, j));
        }
    }

    // --- Tworzenie tensora wejściowego dla ONNX Runtime ---
    vector<int64_t> actual_input_dims = {eigen_input.rows(), eigen_input.cols()};

    MemoryInfo memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Value input_tensor = Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), actual_input_dims.data(), actual_input_dims.size());

    if (!input_tensor.IsTensor()) {
        cerr << "Blad: input_tensor nie jest tensorem." << endl;
        return 1;
    }

    // --- Uruchomienie inferencji ---
    vector<const char*> input_node_names = {input_name};
    vector<const char*> output_node_names = {output_name};

    vector<Value> output_tensors;
    try {
        cout << "Uruchamianie inferencji..." << endl;
        output_tensors = session.Run(RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        cout << "Inferencja zakonczona." << endl;
    } catch (const Ort::Exception& e) {
        cerr << "Blad podczas uruchamiania inferencji: " << e.what() << endl;
        return 1;
    }


    // --- Przetwarzanie wyników ---
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        cerr << "Blad: Wynik inferencji nie jest tensorem lub jest pusty." << endl;
        return 1;
    }

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    vector<int64_t> output_dims = output_tensor_info.GetShape();
    size_t output_data_size = output_tensor_info.GetElementCount();

    cout << "\nKształt tensora wyjsciowego: ";
    for (int64_t dim : output_dims) {
        cout << dim << " ";
    }
    cout << endl;
    cout << "Liczba elementow wyjsciowych: " << output_data_size << endl;


    cout << "Predykcje:" << endl;
    for (size_t i = 0; i < output_data_size; ++i) {
        cout << "Wynik[" << i << "]: " << output_data[i] << endl;
    }

    if (output_dims.size() == 2) {
        MatrixXf eigen_output = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(output_data, output_dims[0], output_dims[1]);
        cout << "\nWynik (Eigen Matrix):\n" << eigen_output << endl;
    }

    cout << "\nProgram zakonczony pomyslnie." << endl;
    return 0;
}

