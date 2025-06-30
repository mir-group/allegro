Accelerations
=============

Allegro provides several acceleration options to optimize performance for both training and inference on GPUs. 
Present acceleration options include our :doc:`CuEquivariance integration <cuequivariance>` or our :doc:`custom Triton kernels <triton>`, which can be activated to accelerate the Allegro tensor product.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35
   :align: center

   * - Feature
     - CuEquivariance
     - Triton
   * - **Training**
     - ✅ Yes
     - ❌ No
   * - **Inference**
     - ✅ Yes
     - ✅ Yes
   * - **Platforms**
     - NVIDIA GPUs
     - NVIDIA and AMD GPUs

.. toctree::
   :hidden:
   :maxdepth: 1
    
   cuequivariance
   triton
