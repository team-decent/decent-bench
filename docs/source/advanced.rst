Advanced Developer Guide
========================

This guide covers the more advanced and unique architectural approaches taken in :doc:`Decent Bench <index>`, 
particularly focusing on the interoperability system that enables seamless framework-agnostic operations.

.. contents:: Contents
   :local:
   :depth: 2

Runtime Array Unwrapping
------------------------

Overview
~~~~~~~~

One of the most unique features of :doc:`Decent Bench <index>` is its **runtime array unwrapping** mechanism. 
The :class:`~decent_bench.utils.array.Array` class serves as a wrapper around framework-specific 
array types (NumPy, PyTorch, TensorFlow, JAX, etc) that provides operator overloading and seamless 
interoperability. To avoid the performance overhead of creating wrapper objects, the interoperability 
system internally unwraps arrays at runtime using Python's :class:`~typing.TYPE_CHECKING` constant, while still 
presenting the :class:`~decent_bench.utils.array.Array` type to users and type checkers.

How It Works
~~~~~~~~~~~~

The key to this approach is the ``decent_bench.utils.interoperability._helpers._return_array``
helper function:

.. code-block:: python

    from typing import TYPE_CHECKING

    def _return_array(array: SupportedArrayTypes) -> Array:
        """
        Wrap a framework-native array in an Array wrapper.
        
        This helper standardizes return types across interoperability functions,
        returning the same framework-native object at runtime, while providing a
        typed Array during static type checking.
        """
        if not TYPE_CHECKING:
            return array  # Return native array at runtime
        
        return Array(array)  # Only for type checkers

**Static Type Checking (Development Time):**

When type checkers like ``mypy`` or ``pyright`` analyze your code, ``TYPE_CHECKING`` is ``True``, 
so they see the function returning :class:`~decent_bench.utils.array.Array` objects. This provides proper type hints and IDE support.

**Runtime Execution:**

When Python actually executes the code, ``TYPE_CHECKING`` is ``False``, so the function directly 
returns the native framework array (:class:`numpy.ndarray`, :class:`torch.Tensor`, etc.) without creating an 
additional wrapper object. This means:

- **Zero overhead**: No wrapper objects are created at runtime
- **Native performance**: Operations execute at full framework speed  
- **Transparent to users**: Users work with :class:`~decent_bench.utils.array.Array` objects via operator overloading and see consistent behavior

Example
~~~~~~~

Consider this interoperability function:

.. code-block:: python

    def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
        """Stack arrays along a new dimension."""
        # Extract native arrays from wrappers
        # Will only be Array objects during type checking or
        # if it is the first time any operation is performed on them
        values = [arr.value if isinstance(arr, Array) else arr 
                  for arr in arrays]
        
        if isinstance(values[0], np.ndarray):
            result = np.stack(values, axis=dim)
            return _return_array(result)  # Returns np.ndarray at runtime!
        # ... other frameworks

Then when you write:

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
    
    # Users create arrays using interoperability functions
    x = iop.randn((3,), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    y = iop.zeros((3,), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    
    # Stack them together
    result = iop.stack([x, y, x + y], dim=0)
    
* **Type checker sees:** result is :class:`~decent_bench.utils.array.Array`
* **Runtime:** result is actually :class:`numpy.ndarray` (unwrapped for performance)
* **Users:** interact with it as an :class:`~decent_bench.utils.array.Array` **through operators**

**Why This Matters:**

1. **Type Safety**: Developers get full IDE support and type checking for :class:`~decent_bench.utils.array.Array` operations
2. **Performance**: Zero runtime overhead - internally uses native arrays without wrapper object creation
3. **Seamless Interoperability**: Users write framework-agnostic code using :class:`~decent_bench.utils.array.Array` objects and operators
4. **Consistency**: The same :class:`~decent_bench.utils.array.Array` interface works across NumPy, PyTorch, TensorFlow, and other supported frameworks

Design Goals and Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Design Goals:**

- **Performance**: Eliminate overhead from wrapper objects while maintaining clean abstraction
- **Interoperability**: Provide a unified interface across NumPy, PyTorch, TensorFlow, etc.
- **Type Safety**: Enable full static type checking and IDE support
- **User Experience**: Users work with :class:`~decent_bench.utils.array.Array` objects; unwrapping is an internal optimization

**How Users Interact:**

Users should work with :class:`~decent_bench.utils.array.Array` objects and rely on:

- **Array creation**: Use :func:`iop.randn() <decent_bench.utils.interoperability.randn>`, :func:`iop.zeros() <decent_bench.utils.interoperability.zeros>`, etc. to create arrays
- **Operator overloading**: Use ``+``, ``-``, ``*``, ``/``, ``@``, etc. on :class:`~decent_bench.utils.array.Array` objects
- **Interoperability functions**: Use :func:`iop.sum() <decent_bench.utils.interoperability.sum>`, :func:`iop.mean() <decent_bench.utils.interoperability.mean>`, :func:`iop.transpose() <decent_bench.utils.interoperability.transpose>`, etc.
- **Framework conversion**: Use :func:`iop.to_torch() <decent_bench.utils.interoperability.to_torch>`, :func:`iop.to_numpy() <decent_bench.utils.interoperability.to_numpy>`, etc. when needed

The fact that arrays are internally unwrapped at runtime is a performance optimization that 
users don't need to think about - they simply work with :class:`~decent_bench.utils.array.Array` objects throughout their code.
Never directly instantiate :class:`~decent_bench.utils.array.Array` objects; **always use the interoperability functions**.


Array Class and Interoperability Package
-----------------------------------------

How They Work Together
~~~~~~~~~~~~~~~~~~~~~~

The :class:`~decent_bench.utils.array.Array` class and the :mod:`~decent_bench.utils.interoperability` 
package work in tandem to provide seamless cross-framework operations:

1. **Array Class**: Provides operator overloading (``+``, ``-``, ``*``, ``@``, etc.)
2. **Interoperability Package**: Implements the actual framework-specific operations
3. **Runtime Unwrapping**: Ensures zero performance overhead

**Complete Example:**

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
    
    # Create Array objects using interoperability functions
    # This ensures frameworks and devices are correctly handled
    x = iop.randn((10, 5), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    y = iop.ones_like(x) # Create an array of ones with same shape/framework/device as x
    weight = iop.randn((5, 3), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    
    # Option 1: Use operators (calls iop functions internally)
    z = x + y           # Calls iop.add(x, y) internally
    z = x @ weight      # Calls iop.matmul(x, weight) internally
    z = z ** 2          # Calls iop.power(z, 2) internally
    
    # Option 2: Use interoperability functions directly
    z = iop.add(x, y)
    z = iop.matmul(x, weight)
    mean = iop.mean(z)
    norm = iop.norm(z)
    
* Both approaches work identically
* Both are framework-agnostic
* Both benefit from runtime unwrapping

**Operator Overloading Implementation:**

The :class:`~decent_bench.utils.array.Array` class delegates all operators to interoperability functions:

.. code-block:: python

    class Array:
        def __add__(self, other):
            return iop.add(self, other)  # Delegates to interop
        
        def __matmul__(self, other):
            return iop.matmul(self, other)  # Delegates to interop

This means users get clean syntax (``x + y``) while the interoperability package handles 
framework detection and native operations in one unified system.


Interoperability System
-----------------------

Architecture
~~~~~~~~~~~~

The interoperability system is designed with several layers:

1. **Type Definitions** (``_imports_types.py``): Conditional imports and type aliases
2. **Helper Functions** (``_helpers.py``): Framework detection and conversion utilities
3. **Core Functions** (``_functions.py``): Array creation, conversion, and manipulation
4. **Operators** (``_operators.py``): Arithmetic and mathematical operations
5. **Extended Operations** (``_ext.py``): In-place operations, extension package meant for advanced use
6. **Decorators** (``_decorators.py``): Automatic type conversion for class methods

Framework Detection
~~~~~~~~~~~~~~~~~~~

The system automatically detects which framework an array belongs to:

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    
    framework, device = iop.framework_device_of_array(my_array)
    # Returns: (SupportedFrameworks.NUMPY, SupportedDevices.CPU)

This is used internally to route operations to the correct framework-specific implementation.


Implementing New Interoperability Functions
--------------------------------------------

If you need to add a new operation to the interoperability layer, follow this pattern:

Step 1: Define the Function Signature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your function in ``decent_bench/utils/interoperability/_functions.py`` (for general operations) 
or ``_operators.py`` (for arithmetic operations). The signature should accept :class:`~decent_bench.utils.array.Array` as input
(for arithmetic operations in ``_operators.py`` also accept :class:`~decent_bench.utils.types.SupportedArrayTypes`) and return :class:`~decent_bench.utils.array.Array`.

.. code-block:: python

    def my_operation(
        array: Array,
        parameter: int,
    ) -> Array:
        """
        Description of your operation.
        
        Args:
            array (Array): Input array.
            parameter (int): Description of parameter.
            
        Returns:
            Result in the same framework type as the input.
            
        Raises:
            TypeError: if the framework type is unsupported.
        """

Step 2: Extract the Native Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always extract the underlying native array first in case the input array is not already unwrapped (happens on first use):

.. code-block:: python

    def my_operation(array: Array, parameter: int) -> Array:
        # Extract native array if wrapped
        value = array.value if isinstance(array, Array) else array

Step 3: Implement Framework-Specific Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``isinstance`` checks to handle each framework:

.. code-block:: python

    def my_operation(array: Array | SupportedArrayTypes, parameter: int) -> Array:
        value = array.value if isinstance(array, Array) else array
        
        # NumPy implementation
        if isinstance(value, np.ndarray | np.generic):
            result = np.my_numpy_function(value, parameter)
            return _return_array(result)
        
        # PyTorch implementation
        if torch and isinstance(value, torch.Tensor):
            result = torch.my_torch_function(value, parameter)
            return _return_array(result)
        
        # TensorFlow implementation
        if tf and isinstance(value, tf.Tensor):
            result = tf.my_tf_function(value, parameter)
            return _return_array(result)
        
        # JAX implementation
        if jnp and isinstance(value, jnp.ndarray | jnp.generic):
            result = jnp.my_jax_function(value, parameter)
            return _return_array(result)
        
        raise TypeError(f"Unsupported framework type: {type(value)}")

**Important Notes:**

- Always check if the framework is imported before using it (``if torch and ...``)
- Use the type tuples from ``_imports_types.py``: ``_np_types``, ``_torch_types``, etc if multiple types are acceptable, see their definition for allowed types.
- Always return using ``_return_array()`` for the unwrapping mechanism to work
- Raise ``TypeError`` with a descriptive message for unsupported types

Step 4: Handle Device Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For operations that create new arrays, use the framework and device parameter:

.. code-block:: python

    def my_creation_function(
        shape: tuple[int, ...],
        framework: SupportedFrameworks,
        device: SupportedDevices,
    ) -> Array:
        # Convert device literal to framework-specific representation
        framework_device = device_to_framework_device(device, framework)
        
        if framework == SupportedFrameworks.TORCH:
            result = torch.my_function(shape, device=framework_device)
            return _return_array(result)
        # ... other frameworks

Step 5: Export Your Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your function to ``decent_bench/utils/interoperability/__init__.py``:

.. code-block:: python

    from ._functions import (
        # ... existing imports
        my_operation,
    )
    
    __all__ = [
        # ... existing exports
        "my_operation",
    ]

Step 6: Add Tests
~~~~~~~~~~~~~~~~~

Create comprehensive tests in ``test/utils/test_interoperability.py``:

.. code-block:: python

    @pytest.mark.parametrize(
        "framework,device",
        [
            (SupportedFrameworks.NUMPY, SupportedDevices.CPU),
            pytest.param(SupportedFrameworks.TORCH, SupportedDevices.CPU, marks=pytest.mark.skipif(
                not TORCH_AVAILABLE, reason="PyTorch not available"
            )),
            # ... other frameworks
        ],
    )
    def test_my_operation(framework, device):
        arr = create_array([1.0, 2.0, 3.0], framework, device)
        result = iop.my_operation(arr, parameter=2)
        
        expected = create_array([...], framework, device)
        assert_arrays_equal(result, expected, framework)


Adding Support for New Frameworks
----------------------------------

If you want to extend :doc:`Decent Bench <index>` to support additional array/tensor frameworks beyond 
the already supported ones, follow this guide.

Overview
~~~~~~~~

Adding a new framework requires changes across multiple files in the interoperability system:

1. Update type definitions and imports
2. Add framework literal to supported frameworks
3. Implement device handling
4. Update framework detection logic
5. Add conversion functions
6. Update all existing interoperability operations
7. Add comprehensive tests

This is a significant undertaking but follows a consistent pattern throughout.

Step 1: Update Type Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``decent_bench/utils/interoperability/_imports_types.py``:

.. code-block:: python

    # Add conditional import for your framework
    myframework = None
    with contextlib.suppress(ImportError, ModuleNotFoundError):
        import myframework as _myframework
        myframework = _myframework
    
    _myframework_types = (
        myframework.Tensor, ... if myframework else (float,),
    )

Step 2: Add Framework Literal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``decent_bench/utils/types.py`` to add your framework to the ``SupportedFrameworks`` enum:

.. code-block:: python

    class SupportedFrameworks(Enum):
        """Supported deep learning frameworks."""
        
        NUMPY = "numpy"
        TORCH = "torch"
        TENSORFLOW = "tensorflow"
        JAX = "jax"
        MYFRAMEWORK = "myframework"  # Add your framework

Step 3: Implement Device Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``decent_bench/utils/interoperability/_helpers.py`` to handle device conversion:

.. code-block:: python

    def device_to_framework_device(
        device: SupportedDevices, 
        framework: SupportedFrameworks
    ) -> Any:
        """Convert SupportedDevices literal to framework-specific device."""
        # ... existing frameworks ...
        
        if myframework and framework == SupportedFrameworks.MYFRAMEWORK:
            # Implement framework-specific device handling
            # Return the appropriate device representation
            if device == SupportedDevices.CPU:
                return myframework.device("cpu")
            return myframework.device("gpu")
        
        raise ValueError(f"Unsupported framework: {framework}")

Update the ``framework_device_of_array`` function:

.. code-block:: python

    def framework_device_of_array(array: Array) -> tuple[SupportedFrameworks, SupportedDevices]:
        """Determine the framework and device of the given Array."""
        value = array.value if isinstance(array, Array) else array
        
        # ... existing framework checks ...
        
        if myframework and isinstance(value, _myframework_types):
            device_str = value.device  # Adjust based on framework API
            device_type = (
                SupportedDevices.GPU if "gpu" in device_str 
                else SupportedDevices.CPU
            )
            return SupportedFrameworks.MYFRAMEWORK, device_type
        
        raise TypeError(f"Unsupported framework type: {type(value)}")

Step 4: Add Conversion Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a conversion function in ``decent_bench/utils/interoperability/_functions.py``:

.. code-block:: python

    # Define type tuple for isinstance checks
    if TYPE_CHECKING:
        ... # existing imports
        from myframework import Tensor as MyFrameworkTensor

    def to_myframework(
        array: Array | SupportedArrayTypes, 
        device: SupportedDevices
    ) -> MyFrameworkTensor:
        """
        Convert input array to a MyFramework tensor.
        
        Args:
            array (Array | SupportedArrayTypes): Input Array
            device (SupportedDevices): Device of the input array.
            
        Returns:
            MyFrameworkTensor: Converted tensor.
            
        Raises:
            ImportError: if MyFramework is not installed.
        """
        if not myframework:
            raise ImportError("MyFramework is not installed.")
        
        value = array.value if isinstance(array, Array) else array
        framework_device = device_to_framework_device(
            device, SupportedFrameworks.MYFRAMEWORK
        )
        
        # Handle conversion from each supported framework
        if isinstance(value, myframework.Tensor):
            return cast("MyFrameworkTensor", value.to(framework_device))
        if isinstance(value, np.ndarray | np.generic):
            return cast("MyFrameworkTensor", 
                       myframework.from_numpy(value).to(framework_device))
        if torch and isinstance(value, torch.Tensor):
            return cast("MyFrameworkTensor",
                       myframework.from_numpy(value.cpu().numpy()).to(framework_device))
        # ... handle other frameworks ...
        
        # Try a direct conversion to check if possible
        return cast("MyFrameworkTensor", 
                   myframework.tensor(value, device=framework_device))

Update the ``to_array`` and all other ``to_"framework"`` functions to include your framework:

.. code-block:: python

    def to_array(
        array: Array | SupportedArrayTypes,
        framework: SupportedFrameworks,
        device: SupportedDevices,
    ) -> Array:
        """Convert an array to the specified framework type."""
        # ... existing frameworks ...
        
        if myframework and framework == SupportedFrameworks.MYFRAMEWORK:
            return _return_array(to_myframework(array, device))
        
        raise TypeError(f"Unsupported framework type: {framework}")

Step 5: Update All Interoperability Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every function in ``_operators.py`` and ``_functions.py`` needs to handle your framework.

Example for an operator in ``_operators.py``:

.. code-block:: python

    def add(array1: Array | SupportedArrayTypes, 
            array2: Array | SupportedArrayTypes) -> Array:
        """Element-wise addition of two arrays."""
        value1 = array1.value if isinstance(array1, Array) else array1
        value2 = array2.value if isinstance(array2, Array) else array2
        
        # ... existing frameworks ...
        
        if myframework and isinstance(value1, _myframework_types):
            return _return_array(myframework.add(value1, value2))
        
        raise TypeError(f"Unsupported framework type: {type(value1)}")

Example for a function in ``_functions.py``:

.. code-block:: python

    def sum(
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Sum elements of an array."""
        value = array.value if isinstance(array, Array) else array
        
        # ... existing frameworks ...
        
        if myframework and isinstance(value, _myframework_types):
            return _return_array(myframework.sum(value, axis=dim, keepdims=keepdims))
        
        raise TypeError(f"Unsupported framework type: {type(value)}")

You'll need to update every operation: ``add``, ``sub``, ``mul``, ``div``, ``matmul``, ``power``, 
``sqrt``, ``mean``, ``max``, ``min``, ``transpose``, ``reshape``, ``zeros``, ``ones``, ``randn``, etc.

Step 6: Update Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``_decorators.py`` to handle conversion in the ``autodecorate_cost_method``:

.. code-block:: python

    def _get_converter(framework: SupportedFrameworks) -> Callable:
        if framework == SupportedFrameworks.NUMPY:
            return to_numpy
        if framework == SupportedFrameworks.TORCH:
            return to_torch
        if framework == SupportedFrameworks.TENSORFLOW:
            return to_tensorflow
        if framework == SupportedFrameworks.JAX:
            return to_jax
        if framework == SupportedFrameworks.MYFRAMEWORK:
            return to_myframework
        
        raise ValueError(f"Unsupported framework: {framework}")

Step 7: Export New Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add exports to ``decent_bench/utils/interoperability/__init__.py``:

.. code-block:: python

    from ._functions import (
        # ... existing imports ...
        to_myframework,
    )
    
    __all__ = [
        # ... existing exports ...
        "to_myframework",
    ]

Step 8: Add Comprehensive Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update tests and add test for `to_myframework` in ``test/utils/test_interoperability.py``:

.. code-block:: python

    # Add availability check
    try:
        import myframework
        MYFRAMEWORK_AVAILABLE = True
        MYFRAMEWORK_GPU_AVAILABLE = myframework.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        MYFRAMEWORK_AVAILABLE = False
        MYFRAMEWORK_GPU_AVAILABLE = False
    
    # Add to parameterized tests
    @pytest.mark.parametrize(
        "framework,device",
        [
            (SupportedFrameworks.NUMPY, SupportedDevices.CPU),
            # ... existing frameworks ...
            pytest.param(SupportedFrameworks.MYFRAMEWORK, SupportedDevices.CPU, marks=pytest.mark.skipif(
                not MYFRAMEWORK_AVAILABLE, reason="MyFramework not available"
            )),
            pytest.param(SupportedFrameworks.MYFRAMEWORK, SupportedDevices.GPU, marks=pytest.mark.skipif(
                not MYFRAMEWORK_GPU_AVAILABLE, reason="MyFramework GPU not available"
            )),
        ],
    )
    def test_to_my_framework(framework, device):
        pass

Step 9: Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update the documentation to mention the new framework:

- Add to the list of supported frameworks in ``docs/source/user.rst``
- Update any framework-specific examples
- Add installation instructions if needed

Checklist
~~~~~~~~~

Use this checklist when adding a new framework:

.. code-block:: text

    ☐ Add conditional import to _imports_types.py
    ☐ Define type tuple (_myframework_types)
    ☐ Add to SupportedFrameworks enum in types.py
    ☐ Implement device_to_framework_device
    ☐ Implement framework_device_of_array detection
    ☐ Implement to_myframework conversion
    ☐ Update to_array function
    ☐ Update to_"framework" for every "framework" to handle your framework
    ☐ Update all operators: add, sub, mul, div, matmul, power, etc.
    ☐ Update all in-place operators: iadd, isub, imul, idiv, ipow
    ☐ Update all functions: sum, mean, min, max, argmax, argmin, etc.
    ☐ Update creation functions: zeros, eye, randn, etc.
    ☐ Update utility functions: shape, reshape, transpose, stack, etc.
    ☐ Update _get_converter in _decorators.py
    ☐ Export to_myframework in __init__.py
    ☐ Add framework availability checks in tests
    ☐ Add to parameterized test fixtures
    ☐ Test all operations with new framework
    ☐ Test CPU and GPU devices (if applicable)
    ☐ Update documentation
    ☐ Add installation instructions

Common Considerations
~~~~~~~~~~~~~~~~~~~~~

**API Differences:**

Different frameworks have different APIs. Pay attention to:

- Parameter names (``axis`` vs ``dim`` vs ``dimension``)
- Return types (some frameworks return scalars, others return 0-d arrays)
- Indexing behavior
- Broadcasting rules
- Gradient computation (some frameworks track gradients by default)

**Performance:**

- Some frameworks may not support certain operations efficiently
- Consider framework-specific optimizations
- Be aware of memory layout differences (row-major vs column-major)

**Device Management:**

- Not all frameworks support GPU computation
- Device transfer may have different APIs
- Some frameworks use different GPU backends (CUDA, ROCm, Metal, etc.)

**Type System:**

- Be careful with dtype conversions
- Some frameworks have more restrictive type systems
- Handle scalar vs array returns consistently

**Main Goals:**

- Mimic NumPy behavior as closely as possible
- Maintain consistent behavior across frameworks
- Ensure performance is acceptable
- Provide clear error messages for unsupported operations


Advanced Decorator: autodecorate_cost_method
---------------------------------------------

Purpose
~~~~~~~

The :func:`~decent_bench.utils.interoperability.autodecorate_cost_method` decorator is a specialized 
decorator that automatically handles type conversion for :class:`~decent_bench.costs.Cost` subclass methods. 
It enables users to implement cost functions in their preferred framework while the decorator handles 
conversion automatically.

How It Works
~~~~~~~~~~~~

The decorator performs three key operations:

1. **Unwraps Input Arrays**: Converts :class:`~decent_bench.utils.array.Array` arguments to the cost's native framework type
2. **Calls the Method**: Executes the user's framework-specific implementation
3. **Wraps Output**: Converts the return value back to :class:`~decent_bench.utils.array.Array` if the superclass expects it (still using runtime unwrapping)

Usage Pattern
~~~~~~~~~~~~~

When implementing a custom cost function:

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    import numpy as np
    from numpy.typing import NDArray
    from decent_bench.costs import Cost
    
    class MyCustomCost(Cost):
        
        @iop.autodecorate_cost_method(Cost.function)
        def function(self, x: NDArray[float]) -> float:
            # Implement using NumPy
            # Decorator handles Array -> NDArray conversion
            return float(np.sum(x ** 2))
        
        @iop.autodecorate_cost_method(Cost.gradient)
        def gradient(self, x: NDArray[float]) -> NDArray[float]:
            # Implement using NumPy
            # Decorator handles Array -> NDArray and NDArray -> Array conversion
            return 2 * x

**Key Points:**

- The first argument **must** be named ``x`` (used to determine target framework)
- Use the framework-specific type hints (``NDArray``, ``torch.Tensor``, etc.)
- The decorator matches the superclass method's return type annotation (make sure to specify the correct superclass method you are decorating)
- Warnings are emitted if input arrays have mismatched frameworks

Framework Mismatch Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an input array's framework differs from the cost's framework, a warning is logged:

.. code-block:: python

    # Cost is configured for PyTorch
    my_cost = MyCustomCost(framework=SupportedFrameworks.TORCH, ...)
    
    # But we pass a NumPy array
    result = my_cost.function(numpy_array)  
    # WARNING: Converting array from framework numpy to torch in method function.
    # This may lead to unexpected behavior or performance issues.


Best Practices
--------------

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use Array Objects**: Work with :class:`~decent_bench.utils.array.Array` objects and leverage operator overloading
2. **Avoid Unnecessary Conversions**: Keep arrays in their framework; only convert when needed
3. **Leverage Runtime Unwrapping**: Trust that the system handles performance internally

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
    
    # Good: Create arrays with iop and use operators
    x = iop.randn((100,), SupportedFrameworks.TORCH, SupportedDevices.GPU)
    weight = iop.ones_like(x)
    matrix = iop.eye(100, SupportedFrameworks.TORCH, SupportedDevices.GPU)
    
    for i in range(1000):
        x = x + weight  # Efficient: uses runtime unwrapping
        x = x @ matrix  # No wrapper overhead
    
    # Also good: Use interoperability functions
    for i in range(1000):
        x = iop.add(x, weight)
        x = iop.matmul(x, matrix)
    
* **Avoid:** Manually extracting .value defeats the abstraction
* **Users:** Should not access ``array.value`` directly

Working with Array Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended Usage:**

.. code-block:: python

    import decent_bench.utils.interoperability as iop
    from decent_bench.utils.types import SupportedFrameworks, SupportedDevices
    
    # Create Array objects using interoperability functions
    x = iop.randn((100, 50), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    weight = iop.randn((50, 10), SupportedFrameworks.NUMPY, SupportedDevices.CPU)
    bias = iop.zeros((10,), SupportedFrameworks.NUMPY, SupportedDevices.CPU)

    # or use ones_like, zeros_like etc
    zero_weight = iop.zeros_like(weight)
    one_bias = iop.ones_like(bias)
    
    # Use operators for arithmetic
    result = (x + 1) * 2 / 3
    result = x @ weight + bias
    
    # Use interoperability functions for operations
    mean_val = iop.mean(x)
    std_val = iop.sqrt(iop.mean((x - mean_val) ** 2))
    normalized = (x - mean_val) / std_val
    
    # Convert frameworks when needed
    torch_version = iop.to_torch(x, SupportedDevices.GPU)

**What to Avoid:**

.. code-block:: python

    import numpy as np
    from decent_bench.utils.array import Array
    
    # Don't create Array objects directly
    x = Array(np.array([1, 2, 3]))
    
    # Don't manually extract .value in user code
    native_array = x.value  # Defeats the abstraction
    
    # Don't bypass the Array interface
    result = np.add(x.value, y.value)  # Use x + y instead
    
    # The Array class is meant to be your interface
    # The runtime unwrapping is an internal optimization
    # Always create arrays through iop functions

Testing Framework-Agnostic Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use parameterized tests to verify all frameworks:

.. code-block:: python

    import pytest
    
    @pytest.mark.parametrize("framework", [SupportedFrameworks.NUMPY, ...])
    def test_my_algorithm(framework):
        if framework == SupportedFrameworks.TORCH and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Create arrays in target framework
        x = create_test_array(framework)
        
        # Test your algorithm
        result = my_algorithm(x)
        
        # Verify results
        assert_correct_framework(result, framework)


Common Pitfalls
~~~~~~~~~~~~~~~

**For Users:**

1. **Don't create Array objects directly**: Use :meth:`iop.randn() <decent_bench.utils.interoperability.randn>`, :meth:`iop.zeros() <decent_bench.utils.interoperability.zeros>`, etc., not ``Array(...)``
2. **Don't access .value directly**: Use the :class:`~decent_bench.utils.array.Array` interface and operators instead
3. **Don't bypass interoperability**: Use ``x + y`` or :meth:`iop.add() <decent_bench.utils.interoperability.add>`, not ``np.add()``
4. **Trust the abstraction**: Runtime unwrapping is automatic; you don't need to manage it

**For Developers Extending the System:**

1. **Incorrect isinstance checks**: Use the type tuples from ``_imports_types.py``
2. **Missing framework availability checks**: Always check ``if torch and ...``
3. **Not extracting .value in interop functions**: Interop functions must extract the native array
4. **Forgetting _return_array()**: Always use ``_return_array()`` for consistent unwrapping and type checking


Further Reading
---------------

- :doc:`API Reference <api/decent_bench.utils.interoperability>`
- :doc:`User Guide <user>` for basic usage
- `Type Checking Documentation <https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING>`_
