Installation
============

You can download `FROSTIE from GitHub <https://github.com/ishan-mishra/FROSTIE>`_
or clone the repository:

.. code-block:: bash
		
   git clone https://github.com/ishan-mishra/FROSTIE


Linux conda environment setup
-----------------------------

FROSTIE currently supports Python versions up to ____. You can create a new 
anaconda environment with, say, the latest version of Python 3.10 via:

.. code-block:: bash

   conda create --name 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.10

Once the basic Python packages are installed in this fresh environment, you
can activate the environment where POSEIDON will dwell:

.. code-block:: bash

   conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘

Then navigate into the 'FROSTIE' directory and install the package via:

.. code-block:: bash
		
   cd FROSTIE
   pip install -e .

..
   Install FROSTIE from PyPI
   ___________________________

   FROSTIE is also available on the PyPI (Python Package Index), and can be installed using the ``pip`` command

   .. code-block:: bash
         
      pip install FROSTIE

   If using this option, you will need to download the the `data folder from FROSTIE's GitHub repository <https://github.com/ishan-mishra/FROSTIE/docs/content/notebooks/data>`_ separately, in order to successfully run the tutorials locally on your computer. 