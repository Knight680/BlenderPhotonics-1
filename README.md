
BlenderPhotonics
========================

-   Author: Yuxuan Zhang (zhang.yuxuan1 at northeastern.edu) and Qianqian Fang (q.fang at neu.edu)
-   License: GNU General Public License version 3 (GPLv3)
-   Version: 0.5
-   Website: <http://mcx.space/BlenderPhotonics>

Introduction
-------------
BlenderPhotonics is a Blender addon to enable 3-D tetrahedral mesh generation (via [Iso2Mesh](http://iso2mesh.sf.net))
and mesh-based Monte Carlo (MMC) photon simulations (via [MMCLAB](http://mcx.space/wiki/?Learn#mmclab)) inside
the Blender environment. Both Iso2Mesh and MMCLAB are executed in GNU Octave, which interoperates with Blender
via the `oct2py` module and the `bpy` Python interface.

BlenderPhotonics supports three processing workflows: 1) converting 3-D Blender objects to region-labeled
tetrahedral meshes and triangular surfaces; 2) converting a volumetric image stored in a NIfTI file to a
multi-labeled tetrahedral mesh, and 3) defining optical properties of each region and a light source to
execute and render MMC simulation results. Each feature can be achieved via a single click on the GUI.

BlenderPhotonics combines the interactive 3-D shape creation/editing and advanced modeling capabilities 
provided by Blender with state-of-the-art Monte Carlo (MC) light simulation techniques and GPU acceleration. 
It uses Blender's user-friendly computer-aided-design (CAD) interface as the front-end to allow creations 
of complex domains, making it easy-to-use for less-experienced users to create sophisticated optical
simulations needed for a wide range of biophotonics applications.


Installation
-------------

1. Install Blender (2.8 or higher) and Octave (5.0 or lower) and add them to the `PATH` environment variable.
   on Ubuntu Linux, this can be done by `sudo apt-get install blender octave`
2. Install Python module `oct2py` for the bundled (built-in) Python inside Blender
    1. This can be done by first identifying the bundled Python by running blender, go to the 
       *Scripting* tab, in the left-middle Console panel, you can see the Python version, for example, is 3.x
    2. Open a terminal, type `python3 --version`, if the printed version is the same as Blender bundled Python 
       version, you may go to Step 2.4
    3. If your system's python3 is different from Blender's built-in version, you need to install the matching
       version via your package management system, such as `sudo apt-get install python3.x` - here "3.x" must
       match what you saw in the Blender's scripting window.
    4. type `sudo python3 -m pip install oct2py` or `sudo python3.x -m pip install oct2py`, this will download and install
       `oct2py` and its dependencies to the system's python folder, and all other users on the same computer can 
       use it. If you just want to install oct2py for your own account, or do not have `sudo`, you can install by
       `python3 -m pip install oct2py --user` or `python3.x -m pip install oct2py --user`. This wil install all modules
       under `~/.local/lib/python3.x/site-package` folder.
3. Download and unzip Iso2Mesh from http://github.com/fangq/iso2mesh to a work folder
4. Download and unzip MMCLAB from http://mcx.space/nightly/ to a work folder
5. Automatically add Iso2Mesh and MMCLAB to your Octave's search path by opening `~/.octaverc` with a text editor
   and type
   ```
   addpath('/path/to/iso2mesh');
   addpath('/path/to/mmclab');
   ```
   Once completed, start octave, and type `which s2m` and `which mmc`, you should see their paths printed.
6. Install BlenderPhotonics in Blender
    1. Download BlenderPhotonics from Github: https://github.com/COTILab/BlenderPhotonics/
    2. Start Blender, select menu **Edit\Preferences\Add-ons**, then click the **Install ...** button, browse
       the downloaded .zip file. Blender will load the addon and show it as **User Interface: BlenderPhotonics**, 
       click on the empty checkbox, this will install and enable "BlenderPhotonics". It may take a few seconds, until
       the checkmark is shown. You can close the Preferences dialog. You should restart blender to use the addon.
       The addon is installed under the folder `~/.config/blender/2.82/scripts/addons/BlenderPhotonics`
    3. Click on the small `<` button next to the x/y/z-axis icon on the right-top of the Layout view to show the 
       "N-Panel", and BlenderPhotonics is shown as a tab at the bottom. Click on it to see the BlenderPhotonics GUI.

