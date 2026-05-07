#define PY_SSIZE_T_CLEAN
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <Python.h>
#include <stdbool.h>

// Raises a Python exception and returns false if code is not hipSuccess.
static bool gpuAssert(hipError_t code, const char *file, int line) {
    if (code == hipSuccess)
        return true;

    const char *prefix = "ave-lang Error [HIP]: ";
    const char *str = hipGetErrorString(code);
    char err[1024] = {0};
    strcat(err, prefix);
    strcat(err, str);
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
    return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define HIP_CHECK_AND_RETURN_NULL(ans)                                        \
    do {                                                                       \
        if (!gpuAssert((ans), __FILE__, __LINE__))                             \
            return NULL;                                                       \
    } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define HIP_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
    do {                                                                       \
        if (!gpuAssert((ans), __FILE__, __LINE__)) {                           \
            PyEval_RestoreThread(_save);                                       \
            return NULL;                                                       \
        }                                                                      \
    } while (0)

static PyObject *loadBinary(PyObject *self, PyObject *args) {
    const char *name;
    const char *data;
    Py_ssize_t data_size;
    int shared;
    int device;
    if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                          &device)) {
        return NULL;
    }

    hipFunction_t fun;
    hipModule_t mod;
    int32_t n_regs = 0;
    int32_t n_spills = 0;
    int32_t n_max_threads = 0;

    // Initialize HIP
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize HIP");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    // Use hipModuleLoadDataEx like the working test
    result = hipModuleLoadDataEx(&mod, data, 0, NULL, NULL);
    if (result != hipSuccess) {
        Py_BLOCK_THREADS;
        PyErr_SetString(PyExc_RuntimeError, "Failed to load HIP module");
        return NULL;
    }

    result = hipModuleGetFunction(&fun, mod, name);
    if (result != hipSuccess) {
        Py_BLOCK_THREADS;
        PyErr_SetString(PyExc_RuntimeError, "Failed to get HIP function");
        return NULL;
    }

    result = hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fun);
    if (result != hipSuccess) {
        n_regs = 0;
    }

    result =
        hipFuncGetAttribute(&n_spills, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
    if (result != hipSuccess) {
        n_spills = 0;
    }

    result = hipFuncGetAttribute(&n_max_threads,
                                 HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun);
    if (result != hipSuccess) {
        hipDeviceProp_t props;
        result = hipGetDeviceProperties(&props, device);
        if (result == hipSuccess) {
            n_max_threads = props.maxThreadsPerBlock;
        } else {
            n_max_threads = 1024;
        }
    }

    Py_END_ALLOW_THREADS;

    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                         n_spills, n_max_threads);
}


static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hip binary into HIP driver"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hip_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_hip_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
