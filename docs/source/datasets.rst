Datasets
========

Currently supported datasets in the library:

.. raw:: html

   <div>
     <table>
       <tr>
         <td style="padding-right: 40px;">

           <strong>Static</strong>
           <ul>
             <li><a href="https://www.matthewtancik.com/nerf">NeRF-Synthetic</a>: bounded</li>
             <li><a href="https://dorverbin.github.io/refnerf/">Ref-NeRF</a>: bounded</li>
             <li><a href="https://research.nvidia.com/labs/toronto-ai/adaptive-shells/">Shelly</a>: bounded</li>
             <li><a href="https://github.com/lioryariv/idr?tab=readme-ov-file">DTU</a>: unbounded</li>
             <li><a href="https://github.com/Totoro97/NeuS?tab=readme-ov-file">BlendedMVS</a>: unbounded</li>
             <li><a href="https://jonbarron.info/mipnerf360/">Mip-NeRF360</a>: unbounded</li>
           </ul>

         </td>
         <td style="padding-left: 40px;">

           <strong>Dynamic</strong>
           <ul>
             <li><a href="https://www.albertpumarola.com/research/D-NeRF/index.html">D-NeRF</a>: semi-monocular, bounded</li>
             <li><a href="https://dynamic3dgaussians.github.io/">PanopticSports</a>: multi-view, bounded</li>
             <li><a href="https://epic-kitchens.github.io/VISOR/">VISOR</a>: monocular, unbounded</li>
             <li><a href="https://kair-bair.github.io/dycheck/">iPhone</a>: monocular, unbounded</li>
             <li><a href="https://github.com/Junyi42/monst3r">MonST3R</a>: monocular, unbounded</li>
           </ul>

         </td>
       </tr>
     </table>
   </div>


Soon to be added (or not tested):

.. raw:: html

   <div>
     <table>
       <tr>
         <td style="padding-right: 40px;">

           <strong>Static</strong>
           <ul>
             <li><a href="https://www.matthewtancik.com/nerf">NeRF-LLFF</a>: unbounded</li>
             <li>...</li>
           </ul>

         </td>
         <td style="padding-left: 40px;">

           <strong>Dynamic</strong>
           <ul>
             <li><a href="https://github.com/facebookresearch/Neural_3D_Video">Neu3D</a>: multi-view, unbounded</li>
             <li><a href="https://github.com/google/nerfies/releases/tag/0.1">Nerfies</a>: semi-monocular, unbounded</li>
             <li><a href="https://github.com/google/hypernerf/releases/tag/v0.1">Hypernerf</a>: semi-monocular, unbounded</li>
             <li><a href="https://gorokee.github.io/jsyoon/dynamic_synth/">Nvidia Dynamic Scene</a>: semi-monocular, unbounded</li>
             <li><a href="https://people.csail.mit.edu/drdaniel/mesh_animation/#data">AMA</a>: multi-view, bounded</li>
             <li><a href="https://github.com/NVlabs/watch-it-move?tab=readme-ov-file#the-wim-dataset">Robots (WIM)</a>: multi-view, bounded</li>
             <li>...</li>
           </ul>

         </td>
       </tr>
     </table>
   </div>

A dataset labelled as "bounded" means that the dataset is limited to a specific volume, while "unbounded" means that the dataset is not limited to a specific volume, and can be extended indefinitely.
A dataset labelled as "monocular" means that the dataset is captured with a single camera (e.g. casual captures), while "multi-view" means that the dataset is captured with multiple synchronized cameras simultaneosly.
A dataset labelled as "semi-monocular" (e.g. D-NeRF) means that the dataset contain either teleporting cameras or quasi-static scenes :cite:t:`gao2022dynamic`.

Download
--------

Download each dataset by running scripts in the `download <https://github.com/autonomousvision/mvdatasets/tree/main/scripts/download>`_ directory (``scripts/download``). The scripts will download the data and save it to the `data` directory.

Configuration
-------------

Each dataset has a individual configuration (e.g. ``BlenderConfig``) that extends more general configuration ``DatasetConfig`` that all datasets share.
Configuration can be overridden by command line arguments or loaded from file.