#define PY_SSIZE_T_CLEAN
#include <cuda.h>
#include <Python.h>
#include <stdbool.h>

// Raises a Python exception and returns false if code is not CUDA_SUCCESS.
static bool gpuAssert(CUresult code, const char *file, int line) {
    if (code == CUDA_SUCCESS)
        return true;

    const char *prefix = "ave-lang Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
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
#define CUDA_CHECK_AND_RETURN_NULL(ans)                                        \
    do {                                                                       \
        if (!gpuAssert((ans), __FILE__, __LINE__))                             \
            return NULL;                                                       \
    } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
    do {                                                                       \
        if (!gpuAssert((ans), __FILE__, __LINE__)) {                           \
            PyEval_RestoreThread(_save);                                       \
            return NULL;                                                       \
        }                                                                      \
    } while (0)

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
    do {                                                                       \
        if ((funcPointer) == NULL) {                                           \
            (funcPointer) = (initializerFunction)();                           \
            if ((funcPointer) == NULL) {                                       \
                return NULL;                                                   \
            }                                                                  \
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
    CUfunction fun;
    CUmodule mod;
    int32_t n_regs = 0;
    int32_t n_spills = 0;
    int32_t n_max_threads = 0;
    // create driver handles
    CUcontext pctx = 0;

    Py_BEGIN_ALLOW_THREADS;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxGetCurrent(&pctx));
    if (!pctx) {
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
            cuDevicePrimaryCtxRetain(&pctx, device));
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuCtxSetCurrent(pctx));
    }

    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuModuleLoadData(&mod, data));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuModuleGetFunction(&fun, mod, name));
    // get allocated registers and spilled registers from the function
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
    n_spills /= 4;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
        &n_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
        &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));
    if (shared > 49152 && shared_optin > 49152) {
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
            cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
        int shared_total, shared_static;
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
            &shared_total,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
            &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
        CUDA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncSetAttribute(
            fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_optin - shared_static));
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
     "Load provided cubin into CUDA driver"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cuda_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cuda_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
