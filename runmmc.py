"""RunMMC - launch mesh-based Monte Carlo (MMC) simulations using domain configured in Blender

* Authors: (c) 2021-2022 Qianqian Fang <q.fang at neu.edu>
           (c) 2021      Yuxuan Zhang <zhang.yuxuan1 at northeastern.edu>
* License: GNU General Public License V3 or later (GPLv3)
* Website: http://mcx.space/bp

To cite this work, please use the below information

@article{BlenderPhotonics2022,
  author = {Yuxuan Zhang and Qianqian Fang},
  title = {{BlenderPhotonics: an integrated open-source software environment for three-dimensional meshing and photon simulations in complex tissues}},
  volume = {27},
  journal = {Journal of Biomedical Optics},
  number = {8},
  publisher = {SPIE},
  pages = {1 -- 23},
  year = {2022},
  doi = {10.1117/1.JBO.27.8.083014},
  URL = {https://doi.org/10.1117/1.JBO.27.8.083014}
}
"""

import bpy
import numpy as np
import os
from .utils import *

g_nphoton=10000
g_tend=5e-9
g_tstep=5e-9
g_method="grid"
g_outputtype="flux"
g_isreflect=True
g_isnormalized=True
g_basisorder=1
g_debuglevel="TP"
g_gpuid="1"
g_colormap ="jet"
g_tool = '1'
enum_tool=[('1','MCX','Run MCX'),
           ('2', 'MMC', 'Run MMC')]


class runmmc(bpy.types.Operator):
    bl_label = 'Run MMC photon simulation'
    bl_description = "Run mesh-based Monte Carlo simulation"
    bl_idname = 'blenderphotonics.runmmc'

    # creat a interface to set uesrs' model parameter.

    bl_options = {"REGISTER", "UNDO"}
    tool: bpy.props.EnumProperty(default=g_tool, name='Light simulation tool', items = enum_tool)
    nphoton: bpy.props.FloatProperty(default=g_nphoton, name="Photon number")
    tend: bpy.props.FloatProperty(default=g_tend,name="Time gate width (s)")
    tstep: bpy.props.FloatProperty(default=g_tstep,name="Time gate step (s)")
    isreflect: bpy.props.BoolProperty(default=g_isreflect,name="Do reflection")
    isnormalized: bpy.props.BoolProperty(default=g_isnormalized,name="Normalize output")
    basisorder: bpy.props.IntProperty(default=g_basisorder,step=1,name="Basis order (0 or 1)")
    method: bpy.props.EnumProperty(default=g_method, name="Raytracer (use elem)", items = [('elem','elem: Saving weight on elements','Saving weight on elements'),('grid','grid: Dual-grid MMC (not supported)','Dual-grid MMC')])
    outputtype: bpy.props.EnumProperty(default=g_outputtype, name="Output quantity", items = [('flux','flux: fluence rate','fluence rate (J/mm^2/s)'),('fluence','fluence: fluence (J/mm^2)','fluence in J/mm^2'),('energy','energy: energy density J/mm^3','energy density J/mm^3')])
    gpuid: bpy.props.StringProperty(default=g_gpuid,name="GPU ID (01 mask,-1=CPU)")
    debuglevel: bpy.props.StringProperty(default=g_debuglevel,name="Debug flag [MCBWDIOXATRPE]")
    colormap: bpy.props.StringProperty(default=g_colormap, name="color scheme")

    def preparemmc(self):
        import jdata as jd

        ## save optical parameters and source source information
        parameters = [] # mu_a, mu_s, n, g
        try:
            obj = bpy.data.objects["Iso2Mesh"]
            for prop in obj.data.keys():
                parameters.append(obj.data[prop].to_list())
        except:
            for obj in bpy.data.objects[0:-1]:
                if(not ("mua" in obj)):
                    continue
                parameters.append([obj["mua"],obj["mus"],obj["g"],obj["n"]])

        obj = bpy.data.objects['source']
        location =  np.array(obj.location);
        bpy.context.object.rotation_mode = 'QUATERNION'
        direction =  np.array(bpy.context.object.rotation_quaternion).tolist();

        def quaternion2euler(direction):
            w, x, y, z = direction
            R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                          [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
                          [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])
            dir = np.dot(R, np.array([[0], [0], [-1]])).transpose()[0].tolist()
            return dir
        dir = quaternion2euler(direction)

        srcparam1=np.array([val for val in obj['srcparam1']])
        srcparam2=np.array([val for val in obj['srcparam2']])

        # originaze cfg
        outputdir = GetBPWorkFolder();
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        import scipy.io
        vol = scipy.io.loadmat(os.path.join(outputdir,'ImageMesh.mat'))['image']
        affine_matrix = scipy.io.loadmat(os.path.join(outputdir,'ImageMesh.mat'))['scale'] #scale is matrix but just take first numbe
        move = np.array([affine_matrix[0,3],affine_matrix[1,3],affine_matrix[2,3]])
        scale = np.array([affine_matrix[0,0],affine_matrix[1,1],affine_matrix[2,2]])
        cfg={'vol':vol, 'prop':parameters,'srctype':obj['srctype'],'srcpos':(location-move)/scale, 'srcdir':dir,
             'srcparam1':srcparam1/np.append(scale,[1]),'srcparam2':srcparam2/np.append(scale,[1]),'nphoton': self.nphoton,
             'srctype':obj["srctype"],'unitinmm': obj['unitinmm']*scale[0],'tstart':0, 'tend':self.tend, 'tstep':self.tstep,
             'isreflect':self.isreflect,'isnormalized':self.isnormalized, 'method':self.method, 'outputtype':self.outputtype,
             'basisorder':self.basisorder, 'debuglevel':self.debuglevel, 'gpuid':self.gpuid}

        # Run mcx
        import pmcx
        jd.save({'prop':parameters,'cfg':cfg}, os.path.join(outputdir,'mmcinfo.json'));
        res = pmcx.run(cfg)
        jd.save(res, os.path.join(outputdir,'mcx_result.json'))
        flux = np.log10(res['flux'][:,:,:,0]/res['stat']['normalizer']+1) # convert -inf to 0 for color
        result = {'flux':flux, 'scale':affine_matrix}
        jd.save(result, os.path.join(outputdir, 'plot_result.json'))


        #remove all object and import all region as one object
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
        bpy.ops.outliner.orphans_purge(do_recursive=True)

        #LoadVolMesh(result,'MCX_result', outputdir, mode='result_view', colormap=self.colormap)

        print('Finshed!, Please change intereaction mode to Weight Paint to see result!')
        print('''If you prefer a perspective effect，please go to edit mode and make sure shading 'Vertex Group Weight' is on.''')

    def execute(self, context):
        print("Begin to run MMC source transport simulation ...")
        self.preparemmc()
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

#
#   Dialog to set meshing properties
#
class setmmcprop(bpy.types.Panel):
    bl_label = "MMC Simulation Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        global g_nphoton, g_tend, g_tstep, g_method,g_outputtype, g_isreflect, g_isnormalized, g_basisorder, g_debuglevel, g_gpuid
        self.layout.operator("object.dialog_operator")
