#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/npy_math.h>
#include "numpy/arrayobject.h"
#include "sublevel_v5.hpp"

static PyObject *SublevelError;

static PyObject *ResultTransform(std::map<int, sublevel::Ans> res, float* values, sublevel::VirtualCloud* cloud) {
    PyObject* result = PyDict_New();
    npy_intp lenght = (npy_intp)res.size();

    PyArrayObject* birth_time = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_FLOAT, 0);
    PyArrayObject* death_time = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_FLOAT, 0);
    PyArrayObject* eat_time = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_FLOAT, 0);
    PyArrayObject* mean_height = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_FLOAT, 0);

    PyArrayObject* death_size = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_INT32, 0);
    PyArrayObject* eat_size = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_INT32, 0);
    PyArrayObject* minimum_num = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_INT32, 0);
    PyArrayObject* saddle_num = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_INT32, 0);
    PyArrayObject* eat_min_num = (PyArrayObject *) PyArray_ZEROS(1, &lenght, NPY_INT32, 0);

    float* birth_time_val = (float*) PyArray_DATA(birth_time);
    float* death_time_val = (float*) PyArray_DATA(death_time);
    float* eat_time_val = (float*) PyArray_DATA(eat_time);
    float* mean_height_val = (float*) PyArray_DATA(mean_height);

    int* death_size_val = (int*) PyArray_DATA(death_size);
    int* eat_size_val = (int*) PyArray_DATA(eat_size);
    int* minimum_num_val = (int*) PyArray_DATA(minimum_num);
    int* saddle_num_val = (int*) PyArray_DATA(saddle_num);
    int* eat_min_num_val = (int*) PyArray_DATA(eat_min_num);
    
    size_t place = 0;
    sublevel::RootedForest tree(cloud->minima_graph_);
    tree.Initilize();

    auto result2 = PyList_New(res.size());

    for (auto& item : res) {
        minimum_num_val[place] = item.first;
        death_size_val[place] = item.second.death_size;
        birth_time_val[place] = values[minimum_num_val[place]];
        saddle_num_val[place] = item.second.death;
        eat_min_num_val[place] = item.second.eat;
        eat_size_val[place] = item.second.eat_size;
        if (item.second.death == -1) {
            death_time_val[place] = NPY_INFINITY;
            eat_time_val[place] = NPY_INFINITY;
            mean_height_val[place] = NPY_INFINITY;
        } else {
            death_time_val[place] = values[saddle_num_val[place]];
            eat_time_val[place] = values[eat_min_num_val[place]];
            mean_height_val[place] = item.second.total_height/death_size_val[place];
        }

        std::vector<int> way;
        tree.GetWay(minimum_num_val[place], eat_min_num_val[place], way);
        npy_intp len = (npy_intp)way.size();
        PyArrayObject* way_array = (PyArrayObject *) PyArray_ZEROS(1, &len, NPY_INT32, 0);
        int* way_val = (int*) PyArray_DATA(way_array);
        for (size_t i = 0; i < way.size(); ++i) {
            way_val[i] = way[i];
        }
        PyList_SetItem(result2, place, (PyObject*)way_array);

        ++place;
    }

    PyDict_SetItemString(result ,"birth value", (PyObject*)birth_time);
    PyDict_SetItemString(result ,"death value", (PyObject*)death_time);
    PyDict_SetItemString(result ,"birth of eating cluster", (PyObject*)eat_time);
    PyDict_SetItemString(result ,"Id of dead minimum", (PyObject*)minimum_num);
    PyDict_SetItemString(result ,"Id of saddle", (PyObject*)saddle_num);
    PyDict_SetItemString(result ,"Id of eating minimum", (PyObject*)eat_min_num);
    PyDict_SetItemString(result ,"Number of point in dead cluster", (PyObject*)death_size);
    PyDict_SetItemString(result ,"Number of point in eating cluster", (PyObject*)eat_size);
    PyDict_SetItemString(result ,"Mean height of point in cluster", (PyObject*)mean_height);

    PyObject *rslt = PyTuple_New(2);
    PyTuple_SetItem(rslt, 0, result);
    PyTuple_SetItem(rslt, 1, result2);

    return rslt;
}

static PyObject* py_grid(PyObject* self, PyObject* args) {
    PyArrayObject *values_array = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &values_array)) {
        PyErr_SetString(SublevelError, "wrong args!");
        return NULL;
    }
    int dim = (int)PyArray_NDIM(values_array);
    std::vector<size_t> shape(dim);
    size_t grid_size = 1;
    for (int i = 0; i < dim; ++i) {
        shape[i] = (size_t)PyArray_SHAPE(values_array)[dim - 1 - i];
        grid_size *= shape[i];
    }
    float* values = (float *)PyArray_DATA(values_array);
    sublevel::GridCloud grid(values, grid_size, shape);
    grid.GetOrder();
    auto res = grid.SublevelHomology();
    return ResultTransform(res, values, &grid);
}

static PyObject* py_grid_shape(PyObject* self, PyObject* args) {
    PyArrayObject *values_array = NULL, *shape_array = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &values_array, &PyArray_Type, &shape_array)) {
        PyErr_SetString(SublevelError, "wrong args!");
        return NULL;
    }
    if ((int)PyArray_NDIM(values_array) != 1 || (int)PyArray_NDIM(shape_array) != 1) {
        PyErr_SetString(SublevelError, "wrong args!");
        return NULL;
    }
    size_t grid_size = 1;
    int dim = ((int *)PyArray_SHAPE(shape_array))[0];
    std::vector<size_t> shape(dim);
    int* sh = (int*)PyArray_DATA(shape_array);
    for (int i = 0; i < dim; ++i) {
        shape[i] = sh[dim - i - 1];
        grid_size *= shape[i];
    }
    float* values = (float *)PyArray_DATA(values_array);
    sublevel::GridCloud grid(values, grid_size, shape);
    grid.GetOrder();
    auto res = grid.SublevelHomology();
    return ResultTransform(res, values, &grid);
}

static PyObject* py_graph(PyObject* self, PyObject* args) {
    PyArrayObject *values_array = NULL, *graph_array = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &values_array, &PyArray_Type, &graph_array)) {
        PyErr_SetString(SublevelError, "wrong args!");
        return NULL;
    }
    if ((int)PyArray_NDIM(values_array) != 1) {
       PyErr_SetString(SublevelError, "Array of values must have dimention 1!");
       return NULL; 
    }
    if ((int)PyArray_NDIM(graph_array) != 2) {
       PyErr_SetString(SublevelError, "Array of graph must have dimention 2!");
       return NULL; 
    }
    if (PyArray_SHAPE(values_array)[0] != PyArray_SHAPE(graph_array)[0]) {
        PyErr_SetString(SublevelError, "Numbers of points of two arrays must be the same!");
        return NULL;
    }
    float* values = (float *)PyArray_DATA(values_array);
    int* graph_ar = (int *)PyArray_DATA(graph_array);
    size_t cloud_size = (size_t)PyArray_SHAPE(graph_array)[0];
    size_t max = (size_t)PyArray_SHAPE(graph_array)[1];
    sublevel::Graph graph(cloud_size);
    for (size_t id = 0; id < cloud_size; ++id) {
        for (size_t neu_id = 0; neu_id < max; ++ neu_id) {
            int neu = *(graph_ar + id * max + neu_id);
            if (neu == -1 || neu == id) {
                continue;
            }
            if (*(values + id) < *(values + neu) || (*(values + id) == *(values + neu) && id < neu)) {
                graph[neu].push_back(id);
            } else {
                graph[id].push_back(neu);
            }
        }
    }
    sublevel::VirtualCloud cloud(values, cloud_size);
    cloud.SetGraph(std::move(graph));
    cloud.GetOrder();
    auto res = cloud.SublevelHomology();
    return ResultTransform(res, values, &cloud);
}

static PyMethodDef module_methods[] = {
    {"grid", py_grid, METH_VARARGS, "This function take numpy array of values of function of grid and calculate zero sublevel homology of function."},
    {"grid_shape", py_grid_shape, METH_VARARGS, "This function take numpy 1 dimentional array of values of function of grid and 1-dimentional numpy array of shape of this grid. It calculate zero sublevel homology of function."},
    {"graph", py_graph, METH_VARARGS, "This function take numpy array of values and numpy array of graph"},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "sublevel",
    "Claculate zero sublevel homology of function",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_sublevel(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    SublevelError = PyErr_NewException("sublevel.error", NULL, NULL);
    Py_INCREF(SublevelError);
    PyModule_AddObject(m, "error", SublevelError);
    return m;
}
