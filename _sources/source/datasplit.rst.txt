DataSplit
-------------------

Given a dataset split, it aggregated data for frames or pixels level indexing.
All data in stored in CPU.

   .. code-block:: python

      # index frames
      data_split = DataSplit(
         cameras=mv_data.get_split("train"),
         nr_sequence_frames=nr_sequence_frames,
         modalities=mv_data.get_split_modalities("train"),
         index_pixels=False
      )


   .. code-block:: python

      # or, index pixels
      data_split = DataSplit(
         cameras=mv_data.get_split("train"),
         nr_sequence_frames=nr_sequence_frames,
         modalities=mv_data.get_split_modalities("train"),
         index_pixels=True
      )

.. automodule:: mvdatasets.datasplit
   :members:
   :undoc-members: